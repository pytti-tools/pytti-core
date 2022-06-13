from pathlib import Path
from os.path import exists as path_exists
import sys
import os

from loguru import logger


from taming.models import cond_transformer, vqgan

from pytti import replace_grad, clamp_with_grad, vram_usage_mode
import torch
from torch.nn import functional as F
from pytti.image_models import EMAImage
from torchvision.transforms import functional as TF
from PIL import Image
from omegaconf import OmegaConf
import urllib.request
from tqdm import tqdm

VQGAN_MODEL = None
VQGAN_NAME = None
VQGAN_IS_GUMBEL = None

# migrate these to config files
VQGAN_MODEL_NAMES = ["imagenet", "coco", "wikiart", "sflckr", "openimages"]
VQGAN_CONFIG_URLS = {
    "imagenet": ["https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1"],
    # "coco": ["https://dl.nmkd.de/ai/clip/coco/coco.yaml"],
    "coco": ["http://batbot.ai/models/VQGAN/coco_first_stage.yaml"],
    "wikiart": [
        "http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml"
    ],
    "sflckr": [
        "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1"
    ],
    "faceshq": [
        "https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT"
    ],
    "openimages": [
        "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
    ],
}
VQGAN_CHECKPOINT_URLS = {
    "imagenet": ["https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1"],
    # "coco": ["https://dl.nmkd.de/ai/clip/coco/coco.ckpt"],
    "coco": ["http://batbot.ai/models/VQGAN/coco_first_stage.ckpt"],
    "wikiart": [
        "http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt"
    ],
    "sflckr": [
        "https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1"
    ],
    "faceshq": [
        "https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt"
    ],
    "openimages": [
        "https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
    ],
}


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    with urllib.request.urlopen(url) as source:
        file_size = int(source.info().get("Content-Length"))

        # Check if file already downloaded
        if os.path.isfile(dest):
            if os.path.getsize(dest) == file_size:
                return True
            else:
                logger.warning(
                    f"WARNING: Pre-existing file at {dest} does not match the download size, overwriting."
                )

        print(f"Downloading {url} to {dest} ({file_size//1024}KB)")

        with open(dest, "wb") as output, tqdm(total=file_size) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

        return os.path.getsize(dest) == file_size


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == "taming.models.vqgan.VQModel":
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = False
    elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
        del parent_model
        gumbel = False
    elif config.model.target == "taming.models.vqgan.GumbelVQ":
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    else:
        raise ValueError(f"unknown model type: {config.model.target}")
    del model.loss
    return model, gumbel


def vector_quantize(x, codebook, fake_grad=True):
    d = (
        x.pow(2).sum(dim=-1, keepdim=True)
        + codebook.pow(2).sum(dim=1)
        - 2 * x @ codebook.T
    )
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class VQGANImage(EMAImage):
    """
    VQGAN latent image representation
    width:  (positive integer) approximate image width in pixels  (will be rounded down to nearest multiple of 16)
    height: (positive integer) approximate image height in pixels (will be rounded down to nearest multiple of 16)
    model:  (VQGAN) vqgan model
    """

    @vram_usage_mode("VQGAN Image")
    def __init__(
        self, width, height, scale=1, model=VQGAN_MODEL, ema_val=0.99, device=None
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if model is None:
            model = VQGAN_MODEL
            if model is None:
                raise RuntimeError(
                    "ERROR: model is None and VQGAN is not initialized loaded"
                )

        if VQGAN_IS_GUMBEL:
            e_dim = 256
            n_toks = model.quantize.n_embed
            vqgan_quantize_embedding = model.quantize.embed.weight
        else:
            e_dim = model.quantize.e_dim
            n_toks = model.quantize.n_e
            vqgan_quantize_embedding = model.quantize.embedding.weight

        f = 2 ** (model.decoder.num_resolutions - 1)
        self.e_dim = e_dim
        self.n_toks = n_toks

        width *= scale
        height *= scale
        # set up parameter dimensions
        toksX, toksY = width // f, height // f
        sideX, sideY = toksX * f, toksY * f
        self.toksX, self.toksY = toksX, toksY

        # we can't use our own vqgan_quantize_embedding yet because the buffer isn't
        # registered, and we can't register the buffer without the value of z

        z = self.rand_latent(vqgan_quantize_embedding=vqgan_quantize_embedding)
        super().__init__(sideX, sideY, z, ema_val)
        self.output_axes = ("n", "s", "y", "x")
        self.lr = 0.15 if VQGAN_IS_GUMBEL else 0.1
        self.latent_strength = 1

        # extract the parts of VQGAN we need
        self.register_buffer(
            "vqgan_quantize_embedding", vqgan_quantize_embedding, persistent=False
        )
        # self.vqgan_quantize_embedding = torch.nn.Parameter(vqgan_quantize_embedding)
        self.vqgan_decode = model.decode
        self.vqgan_encode = model.encode

    def clone(self):
        dummy = VQGANImage(*self.image_shape)
        with torch.no_grad():
            dummy.representation_parameters.set_(self.representation_parameters.clone())
            dummy.accum.set_(self.accum.clone())
            dummy.biased.set_(self.biased.clone())
            dummy.average.set_(self.average.clone())
            dummy.decay = self.decay
        return dummy

    def get_latent_tensor(self, detach=False, device=None):
        if device is None:
            device = self.device
        z = self.representation_parameters
        if detach:
            z = z.detach()
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        return z_q

    @classmethod
    def get_preferred_loss(cls):
        from pytti.LossAug.LatentLossClass import LatentLoss

        return LatentLoss

    def decode(self, z, device=None):
        if device is None:
            device = self.device
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        out = self.vqgan_decode(z_q).add(1).div(2)
        width, height = self.image_shape
        return clamp_with_grad(out, 0, 1)
        # return F.interpolate(clamp_with_grad(out, 0, 1).to(device, memory_format = torch.channels_last), (height, width), mode='nearest')

    @torch.no_grad()
    def encode_image(self, pil_image, device=None, **kwargs):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        self.representation_parameters.set_(z.movedim(1, 3))
        self.reset()

    @torch.no_grad()
    def make_latent(self, pil_image, device=None):
        if device is None:
            device = self.device
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        z_q = (
            vector_quantize(z.movedim(1, 3), self.vqgan_quantize_embedding)
            .movedim(3, 1)
            .to(device)
        )
        return z_q

    @torch.no_grad()
    def encode_random(self):
        self.representation_parameters.set_(self.rand_latent())
        self.reset()

    def rand_latent(self, device=None, vqgan_quantize_embedding=None):
        if device is None:
            device = self.device
        if vqgan_quantize_embedding is None:
            vqgan_quantize_embedding = self.vqgan_quantize_embedding
        n_toks = self.n_toks
        toksX, toksY = self.toksX, self.toksY
        one_hot = F.one_hot(
            torch.randint(n_toks, [toksY * toksX], device=device), n_toks
        ).float()
        z = one_hot @ vqgan_quantize_embedding
        z = z.view([-1, toksY, toksX, self.e_dim])
        return z

    # Why is this a static method? Make it a regular method and kill the globals.
    @staticmethod
    def init_vqgan(model_name, model_artifacts_path, device=None):
        if device is None:
            # device = self.device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global VQGAN_MODEL, VQGAN_NAME, VQGAN_IS_GUMBEL  # uh.... fix this nonsense.
        if VQGAN_NAME == model_name:
            return
        if model_name not in VQGAN_MODEL_NAMES:
            raise ValueError(
                f"VQGAN model {model_name} is not supported. Supported models are {VQGAN_MODEL_NAMES}"
            )
        model_artifacts_path = Path(model_artifacts_path)
        logger.info(model_artifacts_path)
        model_artifacts_path.mkdir(parents=True, exist_ok=True)
        vqgan_config = model_artifacts_path / f"{model_name}.yaml"
        vqgan_checkpoint = model_artifacts_path / f"{model_name}.ckpt"
        logger.debug(vqgan_config)
        logger.debug(vqgan_config.absolute())
        logger.debug(vqgan_checkpoint.absolute())
        logger.debug(vqgan_checkpoint)

        if not vqgan_config.exists():
            logger.warning(
                f"WARNING: VQGAN config file {vqgan_config} not found. Initializing download."
            )

            url = VQGAN_CONFIG_URLS[model_name][0]

            if not _download(url, vqgan_config):
                logger.critical(
                    f"ERROR: VQGAN model {model_name} config failed to download! Please contact model host or find a new one."
                )
                raise FileNotFoundError(f"VQGAN {model_name} config not found")
        # if not path_exists(vqgan_checkpoint):
        if not vqgan_checkpoint.exists():
            logger.warning(
                f"WARNING: VQGAN checkpoint file {vqgan_checkpoint} not found. Initializing download."
            )

            url = VQGAN_CHECKPOINT_URLS[model_name][0]

            if not _download(url, vqgan_checkpoint):
                logger.critical(
                    f"ERROR: VQGAN model {model_name} checkpoint failed to download! Please contact model host or find a new one."
                )
                raise FileNotFoundError(f"VQGAN {model_name} checkpoint not found")

        VQGAN_MODEL, VQGAN_IS_GUMBEL = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        with vram_usage_mode("VQGAN"):
            VQGAN_MODEL = VQGAN_MODEL.to(device)
        VQGAN_NAME = model_name

    @staticmethod
    def free_vqgan():
        global VQGAN_MODEL
        VQGAN_MODEL = None  # should this maybe be `del VQGAN_MODEL` instead?
