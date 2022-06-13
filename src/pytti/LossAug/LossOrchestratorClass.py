from IPython import display
from loguru import logger
from PIL import Image

from pytti.image_models import PixelImage

# from pytti.LossAug import build_loss
from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss
from pytti.Perceptor.Prompt import parse_prompt

from pytti.LossAug.BaseLossClass import Loss
from pytti.LossAug.DepthLossClass import DepthLoss
from pytti.LossAug.EdgeLossClass import EdgeLoss


#################################


LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}


def build_loss(weight_name, weight, name, img, pil_target):
    # from pytti.LossAug import LOSS_DICT

    weight_name, suffix = weight_name.split("_", 1)
    if weight_name == "direct":
        Loss = type(img).get_preferred_loss()
    else:
        Loss = LOSS_DICT[weight_name]
    logger.debug(type(Loss))
    logger.debug(type(img))
    out = Loss.TargetImage(
        f"{weight_name} {name}:{weight}",
        img.image_shape,
        pil_target,
        img_model=img,  # type(img)
    )
    out.set_enabled(pil_target is not None)
    return out


#################################


def configure_init_image(
    init_image_pil: Image.Image,
    restore: bool,
    img: PixelImage,
    params,
    loss_augs,
    embedder,
    prompts,
):

    if init_image_pil is not None:
        if not restore:
            # move these logging statements into .encode_image()
            logger.info("Encoding image...")
            img.encode_image(init_image_pil)
            logger.info("Encoded Image:")
            # pretty sure this assumes we're in a notebook
            display.display(img.decode_image())
        # set up init image prompt
        init_augs = ["direct_init_weight"]
        init_augs = [
            build_loss(
                x,
                params[x],
                f"init image ({params.init_image})",
                img,
                init_image_pil,
            )
            for x in init_augs
            if params[x] not in ["", "0"]
        ]
        loss_augs.extend(init_augs)
        if params.semantic_init_weight not in ["", "0"]:
            semantic_init_prompt = parse_prompt(
                embedder,
                f"init image [{params.init_image}]:{params.semantic_init_weight}",
                init_image_pil,
            )
            prompts[0].append(semantic_init_prompt)
        else:
            semantic_init_prompt = None
    else:
        init_augs, semantic_init_prompt = [], None

    return init_augs, semantic_init_prompt, loss_augs, img, embedder, prompts


def configure_stabilization_augs(img, init_image_pil, params, loss_augs):
    ## NB: in loss orchestrator impl, this begins with an init_image override.
    ## possibly the source of the issue?
    stabilization_augs = [
        "direct_stabilization_weight",
        "depth_stabilization_weight",
        "edge_stabilization_weight",
    ]
    stabilization_augs = [
        build_loss(x, params[x], "stabilization", img, init_image_pil)
        for x in stabilization_augs
        if params[x] not in ["", "0"]
    ]
    loss_augs.extend(stabilization_augs)

    return loss_augs, img, init_image_pil, stabilization_augs


def configure_optical_flows(img, params, loss_augs):

    if params.animation_mode == "Video Source":
        if params.flow_stabilization_weight == "":
            params.flow_stabilization_weight = "0"
        optical_flows = [
            OpticalFlowLoss.TargetImage(
                f"optical flow stabilization (frame {-2**i}):{params.flow_stabilization_weight}",
                img.image_shape,
            )
            for i in range(params.flow_long_term_samples + 1)
        ]
        for optical_flow in optical_flows:
            optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
    elif params.animation_mode == "3D" and params.flow_stabilization_weight not in [
        "0",
        "",
    ]:
        optical_flows = [
            TargetFlowLoss.TargetImage(
                f"optical flow stabilization:{params.flow_stabilization_weight}",
                img.image_shape,
                device="cuda",
            )
        ]
        for optical_flow in optical_flows:
            optical_flow.set_enabled(False)
        loss_augs.extend(optical_flows)
    else:
        optical_flows = []
    # other loss augs
    if params.smoothing_weight != 0:
        loss_augs.append(TVLoss(weight=params.smoothing_weight))

    return img, loss_augs, optical_flows


#######################################


class LossBuilder:

    LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}

    def __init__(self, weight_name, weight, name, img, pil_target):
        self.weight_name = weight_name
        self.weight = weight
        self.name = name
        self.img = img
        self.pil_target = pil_target

    # uh.... should the places this is beind used maybe just use Loss.__init__?
    # TO DO: let's make this a class attribute on something

    @property
    def weight_category(self):
        return self.weight_name.split("_")[0]

    @property
    def loss_factory(self):
        weight_name = self.weight_category
        if weight_name == "direct":
            Loss = type(self.img).get_preferred_loss()
        else:
            Loss = self.LOSS_DICT[weight_name]
        return Loss

    def build_loss(self) -> Loss:
        """
        Given a weight name, weight, name, image, and target image, returns a loss object

        :param weight_name: The name of the loss function
        :param weight: The weight of the loss
        :param name: The name of the loss function
        :param img: The image to be optimized
        :param pil_target: The target image
        :return: The loss function.
        """
        Loss = self.loss_factory
        out = Loss.TargetImage(
            f"{self.weight_category} {self.name}:{self.weight}",
            self.img.image_shape,
            self.pil_target,
        )
        out.set_enabled(self.pil_target is not None)
        return out


def _standardize_null(weight):
    weight = str(weight).strip()
    if weight in ("", "None"):
        weight = "0"
    if float(weight) == 0:
        weight = ""
    return weight


class LossConfigurator:
    """
    Groups together procedures for initializing losses
    """

    def __init__(
        self,
        init_image_pil: Image.Image,
        restore: bool,
        img: PixelImage,
        embedder,
        prompts,
        # params,
        ########
        direct_image_prompts,
        semantic_stabilization_weight,
        init_image,
        semantic_init_weight,
        animation_mode,
        flow_stabilization_weight,
        flow_long_term_samples,
        smoothing_weight,
        ###########
        direct_init_weight,
        direct_stabilization_weight,
        depth_stabilization_weight,
        edge_stabilization_weight,
    ):
        self.init_image_pil = init_image_pil
        self.img = img
        self.embedder = embedder
        self.prompts = prompts

        self.init_augs = []
        self.loss_augs = []
        self.optical_flows = []
        self.last_frame_semantic = None
        self.semantic_init_prompt = None

        # self.params = params
        self.restore = restore

        ### params
        self.direct_image_prompts = direct_image_prompts
        self.semantic_stabilization_weight = _standardize_null(
            semantic_stabilization_weight
        )
        self.init_image = init_image
        self.semantic_init_weight = _standardize_null(semantic_init_weight)
        self.animation_mode = animation_mode
        self.flow_stabilization_weight = _standardize_null(flow_stabilization_weight)
        self.flow_long_term_samples = flow_long_term_samples
        self.smoothing_weight = _standardize_null(smoothing_weight)

        ######
        self.direct_init_weight = _standardize_null(direct_init_weight)
        self.direct_stabilization_weight = _standardize_null(
            direct_stabilization_weight
        )
        self.depth_stabilization_weight = _standardize_null(depth_stabilization_weight)
        self.edge_stabilization_weight = _standardize_null(edge_stabilization_weight)

    def process_direct_image_prompts(self):
        # prompt parsing shouldn't go here.
        self.loss_augs.extend(
            type(self.img)
            .get_preferred_loss()
            .TargetImage(p.strip(), self.img.image_shape, is_path=True)
            for p in self.direct_image_prompts.split("|")
            if p.strip()
        )

    def process_semantic_stabilization(self):
        last_frame_pil = self.init_image_pil
        if not last_frame_pil:
            last_frame_pil = self.img.decode_image()
        self.last_frame_semantic = parse_prompt(
            self.embedder,
            f"stabilization:{self.semantic_stabilization_weight}",
            last_frame_pil,
        )
        self.last_frame_semantic.set_enabled(self.init_image_pil is not None)
        for scene in self.prompts:
            scene.append(self.last_frame_semantic)

    def configure_losses(self):
        if self.init_image_pil is not None:
            self.configure_init_image()
        self.process_direct_image_prompts()
        if self.semantic_stabilization_weight:
            self.process_semantic_stabilization()
        self.configure_stabilization_augs()
        self.configure_optical_flows()
        self.configure_aesthetic_losses()

        return (
            self.loss_augs,
            self.init_augs,
            self.stabilization_augs,
            self.optical_flows,
            self.semantic_init_prompt,
            self.last_frame_semantic,
            self.img,
        )

    def configure_init_image(self):

        if not self.restore:
            # move these logging statements into .encode_image()
            logger.info("Encoding image...")
            self.img.encode_image(self.init_image_pil)
            logger.info("Encoded Image:")
            # pretty sure this assumes we're in a notebook
            display.display(self.img.decode_image())

        ## wrap this for the flexibility that the loop is pretending to provide...
        # set up init image prompt
        if self.direct_init_weight:
            init_aug = LossBuilder(
                "direct_init_weight",
                self.direct_init_weight,
                f"init image ({self.init_image})",
                self.img,
                self.init_image_pil,
            ).build_loss()
            self.loss_augs.append(init_aug)
            self.init_augs.append(init_aug)

        ########
        if self.semantic_init_weight:
            self.semantic_init_prompt = parse_prompt(
                self.embedder,
                f"init image [{self.init_image}]:{self.semantic_init_weight}",
                self.init_image_pil,
            )
            self.prompts[0].append(self.semantic_init_prompt)

    # stabilization
    def configure_stabilization_augs(self):
        d_augs = {
            "direct_stabilization_weight": self.direct_stabilization_weight,
            "depth_stabilization_weight": self.depth_stabilization_weight,
            "edge_stabilization_weight": self.edge_stabilization_weight,
        }
        stabilization_augs = [
            LossBuilder(
                k, v, "stabilization", self.img, self.init_image_pil
            ).build_loss()
            for k, v in d_augs.items()
            if v
        ]
        self.stabilization_augs = stabilization_augs
        self.loss_augs.extend(stabilization_augs)

    def configure_optical_flows(self):
        optical_flows = None

        if self.animation_mode == "Video Source":
            if self.flow_stabilization_weight == "":
                self.flow_stabilization_weight = "0"
            optical_flows = [
                OpticalFlowLoss.TargetImage(
                    f"optical flow stabilization (frame {-2**i}):{self.flow_stabilization_weight}",
                    self.img.image_shape,
                )
                for i in range(self.flow_long_term_samples + 1)
            ]

        elif self.animation_mode == "3D" and self.flow_stabilization_weight:
            optical_flows = [
                TargetFlowLoss.TargetImage(
                    f"optical flow stabilization:{self.flow_stabilization_weight}",
                    self.img.image_shape,
                )
            ]

        if optical_flows is not None:
            for optical_flow in optical_flows:
                optical_flow.set_enabled(False)
            self.loss_augs.extend(optical_flows)

    def configure_aesthetic_losses(self):
        if self.smoothing_weight != 0:
            self.loss_augs.append(TVLoss(weight=self.smoothing_weight))
