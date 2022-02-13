from os.path import exists as path_exists
import sys, subprocess
from loguru import logger

#if not path_exists("./taming-transformers"):
#    raise FileNotFoundError("ERROR: taming-transformers is missing!")
#if "./taming-transformers" not in sys.path:
#    sys.path.append("./taming-transformers")
#else:
#    logger.debug("DEBUG: sys.path already contains ./taming transformers")
from taming.models import cond_transformer, vqgan

from pytti import DEVICE, replace_grad, clamp_with_grad, vram_usage_mode
import torch
from torch.nn import functional as F
from pytti.Image import EMAImage
from torchvision.transforms import functional as TF
from PIL import Image
from omegaconf import OmegaConf

VQGAN_MODEL = None
VQGAN_NAME = None
VQGAN_IS_GUMBEL = None

# migrate these to config files
VQGAN_MODEL_NAMES = ["imagenet", "coco", "wikiart", "sflckr", "openimages"]
VQGAN_CONFIG_URLS = {
    "imagenet": [
        "curl -L -o imagenet.yaml -C - https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1"
    ],
    "coco": ["curl -L -o coco.yaml -C - https://dl.nmkd.de/ai/clip/coco/coco.yaml"],
    "wikiart": [
        "curl -L -o wikiart.yaml -C - http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml"
    ],
    "sflckr": [
        "curl -L -o sflckr.yaml -C - https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1"
    ],
    "faceshq": [
        "curl -L -o faceshq.yaml -C - https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT"
    ],
    "openimages": [
        "curl -L -o openimages.yaml -C - https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1"
    ],
}
VQGAN_CHECKPOINT_URLS = {
    "imagenet": [
        "curl -L -o imagenet.ckpt -C - https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1"
    ],
    "coco": ["curl -L -o coco.ckpt -C - https://dl.nmkd.de/ai/clip/coco/coco.ckpt"],
    "wikiart": [
        "curl -L -o wikiart.ckpt -C - http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt"
    ],
    "sflckr": [
        "curl -L -o sflckr.ckpt -C - https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1"
    ],
    "faceshq": [
        "curl -L -o faceshq.ckpt -C - https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt"
    ],
    "openimages": [
        "curl -L -o openimages.ckpt -C - https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1"
    ],
}


def load_vqgan_model(config_path, checkpoint_path):
    '''
    Loads a model from a config file and a checkpoint file
    
    :param config_path: Path to the config file
    :param checkpoint_path: The path to the checkpoint file
    :return: The model and whether or not it uses gumbel softmax.
    '''
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
    '''
    Takes a tensor x and a codebook, and returns a quantized version of x
    
    :param x: The input to be quantized
    :param codebook: the codebook of shape (n_code, n_dim)
    :param fake_grad: If True, the gradient of the output with respect to the input will be replaced
    with the identity function, defaults to True (optional)
    :return: The quantized vector
    '''
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
    def __init__(self, width, height, scale=1, model=VQGAN_MODEL, ema_val=0.99):
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
        '''
        Create a new VQGANImage object and set its tensor to a clone of the original object's tensor
        :return: The clone function returns a new VQGANImage object with the same parameters as the
        original.
        '''
        dummy = VQGANImage(*self.image_shape)
        with torch.no_grad():
            dummy.tensor.set_(self.tensor.clone())
            dummy.accum.set_(self.accum.clone())
            dummy.biased.set_(self.biased.clone())
            dummy.average.set_(self.average.clone())
            dummy.decay = self.decay
        return dummy

    def get_latent_tensor(self, detach=False, device=DEVICE):
        '''
        Given a tensor, quantize it using vector quantization and return the quantized tensor
        
        :param detach: if True, the latent tensor is detached from the graph, defaults to False
        (optional)
        :param device: The device to run the model on
        :return: The latent tensor quantized to the nearest neighbor in the embedding.
        '''
        z = self.tensor
        if detach:
            z = z.detach()
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        return z_q

    @classmethod
    def get_preferred_loss(cls):
        '''
        Given a class, return the preferred loss function
        
        :param cls: The class of the loss function
        :return: A class object
        '''
        from pytti.LossAug import LatentLoss

        return LatentLoss

    def decode(self, z, device=DEVICE):
        '''
        Takes a latent vector and converts it into an image.
        
        :param z: The latent vector that we want to decode
        :param device: the device to use for training
        :return: The decoded image.
        '''
        z_q = vector_quantize(z, self.vqgan_quantize_embedding).movedim(3, 1).to(device)
        out = self.vqgan_decode(z_q).add(1).div(2)
        width, height = self.image_shape
        return clamp_with_grad(out, 0, 1)
        # return F.interpolate(clamp_with_grad(out, 0, 1).to(device, memory_format = torch.channels_last), (height, width), mode='nearest')

    @torch.no_grad()
    def encode_image(self, pil_image, device=DEVICE, **kwargs):
        '''
        1. resize the image to the desired size
        2. convert the image to a tensor
        3. encode the image using the VQ-VAE
        4. move the z vector to the GPU
        5. reset the hidden state of the LSTM
        
        :param pil_image: The image to encode
        :param device: The device to run the model on
        '''
        pil_image = pil_image.resize(self.image_shape, Image.LANCZOS)
        pil_image = TF.to_tensor(pil_image)
        z, *_ = self.vqgan_encode(pil_image.unsqueeze(0).to(device) * 2 - 1)
        self.tensor.set_(z.movedim(1, 3))
        self.reset()

    @torch.no_grad()
    def make_latent(self, pil_image, device=DEVICE):
        '''
        Given an image, resize it to the desired image shape, convert it to a tensor, and encode it with
        the VQ-VAE
        
        :param pil_image: The image to be encoded
        :param device: The device to run the model on
        :return: The latent vector z_q
        '''
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
        '''
        Set the tensor to a random latent vector
        '''
        self.tensor.set_(self.rand_latent())
        self.reset()

    def rand_latent(self, device=DEVICE, vqgan_quantize_embedding=None):
        '''
        Generate a random latent vector of shape [toksY, toksX, e_dim]
        
        # Python
        def rand_latent_softmax(self, device=DEVICE, vqgan_quantize_embedding=None):
                if vqgan_quantize_embedding is None:
                    vqgan_quantize_embedding = self.vqgan_quantize_embedding
                n_toks = self.n_toks
                toksX, toksY = self.toksX, self.toksY
                one_hot = F.softmax(
                    torch.randn(toksY * toksX, n_toks, device=device), dim=1
                ).float()
                z = one_hot @ vqgan_quantize_embedding
                z = z.view([-1, toksY
        
        uh... that was an automated docstring. fascinating.
        
        :param device: the device to run the model on
        :param vqgan_quantize_embedding: The embedding matrix that we use to quantize the latent space
        :return: The latent vector
        '''
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

    @staticmethod
    def init_vqgan(model_name, device=DEVICE):
        '''
        Loads the VQGAN model from the config and checkpoint files
        
        :param model_name: The name of the model to use
        :param device: The device to run the model on
        :return: The model, and whether it is a Gumbel model.
        '''
        global VQGAN_MODEL, VQGAN_NAME, VQGAN_IS_GUMBEL
        if VQGAN_NAME == model_name:
            return
        if model_name not in VQGAN_MODEL_NAMES:
            raise ValueError(
                f"VQGAN model {model_name} is not supported. Supported models are {VQGAN_MODEL_NAMES}"
            )
        vqgan_config = f"{model_name}.yaml"
        vqgan_checkpoint = f"{model_name}.ckpt"
        if not path_exists(vqgan_config):
            logger.warning(
                f"WARNING: VQGAN config file {vqgan_config} not found. Initializing download."
            )
            command = VQGAN_CONFIG_URLS[model_name][0].split(" ", 6)
            subprocess.run(command)
            if not path_exists(vqgan_config):
                logger.critical(
                    f"ERROR: VQGAN model {model_name} config failed to download! Please contact model host or find a new one."
                )
                raise FileNotFoundError(f"VQGAN {model_name} config not found")
        if not path_exists(vqgan_checkpoint):
            logger.warning(
                f"WARNING: VQGAN checkpoint file {vqgan_checkpoint} not found. Initializing download."
            )
            command = VQGAN_CHECKPOINT_URLS[model_name][0].split(" ", 6)
            subprocess.run(command)
            if not path_exists(vqgan_checkpoint):
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
        '''
        Releases the global VQGAN model from memory
        '''
        global VQGAN_MODEL
        VQGAN_MODEL = None # should this maybe be `del VQGAN_MODEL` insetad?
