from IPython import display
from loguru import logger
from PIL import Image

from pytti.image_models import PixelImage, RGBImage

from pytti.LossAug import TVLoss, HSVLoss, OpticalFlowLoss, TargetFlowLoss
from pytti.Perceptor.Prompt import parse_prompt
from pytti.eval_tools import parse_subprompt


from pytti.LossAug.BaseLossClass import Loss
from pytti.LossAug.DepthLossClass import DepthLoss
from pytti.LossAug.EdgeLossClass import EdgeLoss


#################################

import torch

LOSS_DICT = {"edge": EdgeLoss, "depth": DepthLoss}


def build_loss(
    weight_name: str,
    weight: str,
    name: str,
    img: RGBImage,
    pil_target: Image,
    device=None,
):
    if device is None:
        device = img.device

    weight_name, suffix = weight_name.split("_", 1)
    if weight_name == "direct":
        loss = type(img).get_preferred_loss()
    else:
        loss = LOSS_DICT[weight_name]

    if pil_target is not None:
        resized = pil_target.resize(img.image_shape, Image.LANCZOS)
        comp = loss.make_comp(resized, device=device)
    else:
        # comp = loss.get_default_comp()
        comp = torch.zeros(1, 1, 1, 1, device=device)
    out = loss(
        comp=comp,
        weight=weight,
        name=f"{weight_name} {name} (direct)",
        image_shape=img.image_shape,
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
    logger.debug(params.device)
    _device = params.device
    optical_flows = []
    if params.animation_mode == "Video Source":
        if params.flow_stabilization_weight == "":
            params.flow_stabilization_weight = "0"
        # TODO: if flow stabilization weight is 0, shouldn't this next block just get skipped?

        for i in range(params.flow_long_term_samples + 1):
            optical_flow = OpticalFlowLoss(
                comp=torch.zeros(1, 1, 1, 1, device=_device),  # ,device=DEVICE)
                weight=params.flow_stabilization_weight,
                name=f"optical flow stabilization (frame {-2**i}) (direct)",
                image_shape=img.image_shape,
                device=_device,
            )  # , device=device)
            optical_flow.set_enabled(False)
            optical_flows.append(optical_flow)

    elif params.animation_mode == "3D" and params.flow_stabilization_weight not in [
        "0",
        "",
    ]:
        optical_flow = TargetFlowLoss(
            comp=torch.zeros(1, 1, 1, 1, device=_device),
            weight=params.flow_stabilization_weight,
            name="optical flow stabilization (direct)",
            image_shape=img.image_shape,
            device=_device,
        )
        optical_flow.set_enabled(False)
        optical_flows.append(optical_flow)

    loss_augs.extend(optical_flows)

    # this shouldn't be in this function based on the name.
    # other loss augs
    if params.smoothing_weight != 0:
        loss_augs.append(
            TVLoss(weight=params.smoothing_weight)
        )  # , device=params.device))

    return img, loss_augs, optical_flows


def _standardize_null(weight):
    weight = str(weight).strip()
    if weight in ("", "None"):
        weight = "0"
    if float(weight) == 0:
        weight = ""
    return weight
