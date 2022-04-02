from IPython import display
from loguru import logger
from PIL import Image

from pytti.Image import PixelImage
from pytti.LossAug import build_loss
from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss
from pytti.Perceptor.Prompt import parse_prompt


class LossOrchestrator:
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
        params,
    ):
        self.init_image_pil = init_image_pil
        self.img = img
        self.embedder = embedder
        self.prompts = prompts
        self.loss_augs = []

        self.params = params
        self.restore = restore

    def process_direct_image_prompts(self):
        # prompt parsing shouldn't go here.
        self.loss_augs.extend(
            type(self.img)
            .get_preferred_loss()
            .TargetImage(p.strip(), self.img.image_shape, is_path=True)
            for p in self.params.direct_image_prompts.split("|")
            if p.strip()
        )

    def process_semantic_stabilization(self):
        params = self.params
        embedder = self.embedder
        init_image_pil = self.init_image_pil
        img = self.img
        # need to add tests for this I think
        if params.semantic_stabilization_weight not in ["0", ""]:
            last_frame_semantic = parse_prompt(
                embedder,
                f"stabilization:{params.semantic_stabilization_weight}",
                init_image_pil if init_image_pil else img.decode_image(),
            )
            last_frame_semantic.set_enabled(init_image_pil is not None)
            for scene in prompts:
                scene.append(last_frame_semantic)
        else:
            last_frame_semantic = None
        self.last_frame_semantic = last_frame_semantic

    def configure_losses(self):
        self.configure_init_image()
        self.process_direct_image_prompts()
        self.process_semantic_stabilization()
        self.configure_stabilization_augs()
        self.configure_optical_flows()

        return (
            self.loss_augs,
            self.init_augs,
            self.optical_flows,
            self.semantic_init_prompt,
            self.last_frame_semantic,
            self.img,
        )

    def configure_init_image(self):
        init_image_pil = self.init_image_pil
        restore = self.restore
        img = self.img
        params = self.params
        loss_augs = self.loss_augs
        embedder = self.embedder
        prompts = self.prompts

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

        (
            self.init_augs,
            self.semantic_init_prompt,
            self.loss_augs,
            self.img,
        ) = (init_augs, semantic_init_prompt, loss_augs, img)

    # stabilization
    def configure_stabilization_augs(self):
        (img, init_image_pil, params, loss_augs) = (
            self.img,
            self.init_image_pil,
            self.params,
            self.loss_augs,
        )

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

        (self.loss_augs, self.img, self.init_image_pil) = (
            loss_augs,
            img,
            init_image_pil,
        )
        # return loss_augs, img, init_image_pil

    def configure_optical_flows(self):
        (img, params, loss_augs) = (self.img, self.params, self.loss_augs)

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

        (self.img, self.loss_augs, self.optical_flows) = (img, loss_augs, optical_flows)
