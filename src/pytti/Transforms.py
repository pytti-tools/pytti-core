import glm, gc, torch, math, os
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np
from pytti import parametric_eval
from pytti.LossAug.DepthLossClass import DepthLoss

# from pytti.Image.PixelImage import PixelImage
from adabins.infer import InferenceHelper  # Not used here

# TB_LOGDIR = "logs"  # to do: make this more easily configurable
from loguru import logger

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(TB_LOGDIR)

PADDING_MODES = {
    "mirror": "reflection",
    "smear": "border",
    "black": "zeros",
    "wrap": "zeros",
}


@torch.no_grad()
def apply_grid(tensor, grid, border_mode, sampling_mode):
    height, width = tensor.shape[-2:]
    if border_mode == "wrap":
        max_offset = torch.max(grid.clamp(min=1))
        min_offset = torch.min(grid.clamp(max=-1))
        max_coord = max(max_offset, abs(min_offset))
        if max_coord > 1:
            mod_offset = int(math.ceil(max_coord))
            # make it odd for sure
            mod_offset += 1 - (mod_offset % 2)
            grid = grid.add(mod_offset).remainder(2.0001).sub(1)
    return F.grid_sample(
        tensor,
        grid,
        mode=sampling_mode,
        align_corners=True,
        padding_mode=PADDING_MODES[border_mode],
    )


@torch.no_grad()
def apply_flow(img, flow, border_mode="mirror", sampling_mode="bilinear", device=None):
    from pytti.image_models.pixel import PixelImage
    from pytti.image_models.rgb_image import RGBImage

    if device is None:
        device = img.device

    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):

        # try:
        tensor = img.get_image_tensor().unsqueeze(0)
        # fallback = False

    # except NotImplementedError:
    else:
        # TODO: move this to VQGANImage.get_image_tensor()
        tensor = TF.to_tensor(img.decode_image()).unsqueeze(0).to(device)
        # fallback = True

    height, width = flow.shape[-2:]
    identity = torch.eye(3).to(device)
    identity = identity[0:2, :].unsqueeze(0)  # for batch
    uv = F.affine_grid(identity, tensor.shape, align_corners=True)
    flow = (
        TF.resize(flow, tensor.shape[-2:])
        .movedim(1, 3)
        .div(torch.tensor([[[[width / 2, height / 2]]]], device=device))
    )
    grid = uv - flow
    tensor = apply_grid(tensor, grid, border_mode, sampling_mode)

    # if not fallback:
    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):
        img.set_image_tensor(tensor.squeeze(0))
        tensor_out = img.decode_tensor().detach()
    else:
        # TODO: move this to VQGANImage.set_image_tensor? .decode_tensor()?
        array = (
            tensor.squeeze()
            .movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))
        tensor_out = tensor.detach()
    return tensor_out


@torch.no_grad()
def zoom_2d(
    img,
    translate=(0, 0),
    zoom=(0, 0),
    rotate=0,
    border_mode="mirror",
    sampling_mode="bilinear",
    device=None,
):

    from pytti.image_models.pixel import PixelImage
    from pytti.image_models.rgb_image import RGBImage

    if device is None:
        device = img.device

    # try:
    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):
        tensor = img.get_image_tensor().unsqueeze(0)
    #    fallback = False
    # except NotImplementedError:
    else:
        tensor = TF.to_tensor(img.decode_image()).unsqueeze(0).to(device)
    #    fallback = True

    height, width = tensor.shape[-2:]
    zy, zx = ((height - zoom[1]) / height, (width - zoom[0]) / width)
    ty, tx = (translate[1] * 2 / height, -translate[0] * 2 / width)
    theta = math.radians(rotate)
    affine = (
        torch.tensor(
            [
                [zx * math.cos(theta), -zy * math.sin(theta), tx],
                [zx * math.sin(theta), zy * math.cos(theta), ty],
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    grid = F.affine_grid(affine, tensor.shape, align_corners=True)
    tensor = apply_grid(tensor, grid, border_mode, sampling_mode)

    # if not fallback:
    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):
        img.set_image_tensor(tensor.squeeze(0))
    else:
        array = (
            tensor.squeeze()
            .movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))
    return img.decode_image()


@torch.no_grad()
def render_image_3d(
    image,
    depth,
    P,
    T,
    border_mode,
    sampling_mode,
    stabilize,
    device=None,
):
    """
    image: n x h x w pytorch Tensor: the image tensor
    depth: h x w pytorch Tensor: the depth tensor
    P: 4 x 4 pytorch Tensor: the perspective matrix
    T: 4 x 4 pytorch Tensor: the camera move matrix
    """
    # create grid of points matching image
    h, w = image.shape[-2:]
    f = w / h
    image = image.unsqueeze(0)

    if device is None:
        device = image.device
    logger.debug(device)
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-f, f, w))
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    xy = torch.cat([x, y], dim=1).to(device)

    # v,u = torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w))
    # u = u.unsqueeze(0).unsqueeze(0)
    # v = v.unsqueeze(0).unsqueeze(0)
    # uv = torch.cat([u,v],dim=1).to(device)
    identity = torch.eye(3).to(device)
    identity = identity[0:2, :].unsqueeze(0)  # for batch
    uv = F.affine_grid(identity, image.shape, align_corners=True)
    # get the depth at each point
    depth = depth.unsqueeze(0).unsqueeze(0)
    # depth = depth.to(device)

    view_pos = torch.cat([xy, -depth, torch.ones_like(depth)], dim=1)
    # view_pos = view_pos.to(device)
    # apply the camera move matrix
    next_view_pos = torch.tensordot(T.float(), view_pos.float(), ([0], [1])).movedim(
        0, 1
    )

    # apply the perspective matrix
    clip_pos = torch.tensordot(P.float(), view_pos.float(), ([0], [1])).movedim(0, 1)
    clip_pos = clip_pos / (clip_pos[:, 3, ...].unsqueeze(1))

    next_clip_pos = torch.tensordot(
        P.float(), next_view_pos.float(), ([0], [1])
    ).movedim(0, 1)
    next_clip_pos = next_clip_pos / (next_clip_pos[:, 3, ...].unsqueeze(1))

    # get the offset
    offset = (next_clip_pos - clip_pos)[:, 0:2, ...]
    # flow_forward = offset.mul(torch.tensor([w/2,h/(2*f)],device = device).view(1,2,1,1))
    # offset[:,1,...] *= -1
    # offset = offset.to(device)
    # render the image
    if stabilize:
        advection = offset.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        offset = offset - advection
    offset = offset.permute(0, 2, 3, 1)
    grid = uv - offset
    # grid = grid.permute(0,2,3,1)
    # grid = grid.to(device)

    return apply_grid(image, grid, border_mode, sampling_mode).squeeze(
        0
    ), offset.squeeze(0)


@torch.no_grad()
def zoom_3d(
    img,
    translate=(0, 0, 0),
    rotate=0,
    fov=45,
    near=180,
    far=15000,
    border_mode="mirror",
    sampling_mode="bilinear",
    stabilize=False,
    device=None,
):

    from pytti.image_models.pixel import PixelImage
    from pytti.image_models.rgb_image import RGBImage

    if device is None:
        device = img.device

    width, height = img.image_shape
    px = 2 / height
    alpha = math.radians(fov)
    depth = 1 / (math.tan(alpha / 2))

    pil_image = img.decode_image()

    # pil_image = pil_image.filter(ImageFilter.GaussianBlur(img.scale))
    f = width / height

    # convert depth map
    depth_map, depth_resized = DepthLoss.get_depth(pil_image, device=device)
    depth_min = np.min(depth_map)  # gets overwritten, unnecessary op
    depth_max = np.max(depth_map)  # gets overwritten, unnecessary op
    # depth_image = Image.fromarray(np.array(np.interp(depth_map.squeeze(), (depth_min, depth_max), (0,255)), dtype=np.uint8))
    depth_map = np.interp(depth_map, (1e-3, 10), (near * px, far * px))
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)

    depth_median = np.median(depth_map.flatten())
    depth_mean = np.mean(depth_map)
    r = depth_min / px
    R = depth_max / px
    mu = (depth_mean + depth_median) / (2 * px)
    ########################################
    logger.debug(f"depth range: {r} (r) to {R} (R)")
    logger.debug(f"mu = {mu}")
    translate = [parametric_eval(x, r=r, R=R, mu=mu) for x in translate]
    rotate = parametric_eval(rotate, r=r, R=R, mu=mu)
    logger.debug(f"moving: {translate}")

    # where to get global_step form?
    # maybe track iterations on a centralized object?
    # _dx, _dy, _dz = translate
    # writer.add_scalars(
    #    main_tag = 'zoom_3d/depth',
    #    tag_scalar_dict = {
    #        'r':r,
    #        'R':R,
    #        'mu':mu
    #    }
    # )

    # writer.add_scalars(
    #    main_tag = 'zoom_3d/translation',
    #    tag_scalar_dict = {
    #        'dx':_dx,
    #        'dy':_dy,
    #        'dz':_dz
    #    }
    # )

    ########################################
    logger.debug(device)
    # try:
    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):
        image_tensor = img.get_image_tensor().to(device)
        depth_tensor = (
            TF.resize(
                torch.from_numpy(depth_map),
                image_tensor.shape[-2:],
                interpolation=TF.InterpolationMode.BICUBIC,
            )
            .squeeze()
            .to(device)
        )
        # fallback = False
    # except NotImplementedError:
    else:
        # fallback path
        image_tensor = TF.to_tensor(pil_image).to(device)
        if depth_resized:
            depth_tensor = (
                TF.resize(
                    torch.from_numpy(depth_map),
                    image_tensor.shape[-2:],
                    interpolation=TF.InterpolationMode.BICUBIC,
                )
                .squeeze()
                .to(device)
            )
        else:
            depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
        # fallback = True

    p_matrix = torch.as_tensor(glm.perspective(alpha, f, 0.1, 4).to_list()).to(device)
    tx, ty, tz = translate
    r_matrix = glm.mat4_cast(glm.quat(*rotate))
    t_matrix = glm.translate(glm.mat4(1), glm.vec3(tx * px, -ty * px, tz * px))

    T_matrix = torch.as_tensor((r_matrix @ t_matrix).to_list()).to(device)
    new_image, flow = render_image_3d(
        image_tensor,
        depth_tensor,
        p_matrix,
        T_matrix,
        border_mode=border_mode,
        sampling_mode=sampling_mode,
        stabilize=stabilize,
        device=device,
    )
    logger.debug(new_image.device)

    # if not fallback:
    if any(isinstance(img, im_type) for im_type in (PixelImage, RGBImage)):
        img.set_image_tensor(new_image)
    else:
        # fallback path
        array = (
            new_image.movedim(0, -1)
            .mul(255)
            .clamp(0, 255)
            .cpu()
            .detach()
            .numpy()
            .astype(np.uint8)[:, :, :]
        )
        img.encode_image(Image.fromarray(array))

    flow_out = flow.div(2).mul(torch.tensor([[[[width, height]]]], device=device))
    return flow_out, img.decode_image()


def animate_2d(
    translate_y,
    translate_x,
    rotate_2d,
    zoom_x_2d,
    zoom_y_2d,
    infill_mode,
    sampling_mode,
    img,
    writer=None,
    i=0,
    t=-1,
):
    for v in (translate_y, translate_x, rotate_2d, zoom_x_2d, zoom_y_2d):
        logger.debug(v)
    tx, ty = parametric_eval(translate_x), parametric_eval(translate_y)
    theta = parametric_eval(rotate_2d)
    logger.debug(f"translating: {tx}, {ty}")
    logger.debug(f"rotating: {theta}")
    zx, zy = parametric_eval(zoom_x_2d), parametric_eval(zoom_y_2d)
    logger.debug(f"zooming: {tx}, {ty}")
    next_step_pil = zoom_2d(
        img,
        (tx, ty),
        (zx, zy),
        theta,
        border_mode=infill_mode,
        sampling_mode=sampling_mode,
    )
    ################
    if writer is not None:
        for k, v in {
            "tx": tx,
            "ty": ty,
            "theta": theta,
            "zx": zx,
            "zy": zy,
            "t": t,
        }.items():
            writer.add_scalar(tag=f"translation_2d/{k}", scalar_value=v, global_step=i)

    return next_step_pil


def animate_video_source(
    i,
    img,
    video_frames,
    optical_flows,
    base_name,
    pre_animation_steps,
    frame_stride,
    steps_per_frame,
    file_namespace,
    reencode_each_frame,
    lock_palette,
    save_every,
    ##
    infill_mode,
    sampling_mode,
    device=None,
):
    # ugh this is GROSSSSS....
    from pytti.image_models.pixel import PixelImage

    # current frame index
    frame_n = min(
        (i - pre_animation_steps) * frame_stride // steps_per_frame,
        len(video_frames) - 1,
    )

    # Next frame index (for forrward flow)
    next_frame_n = min(frame_n + frame_stride, len(video_frames) - 1)

    # get next frame of video
    # * will be used as init_image
    # * will be used to estimate forward flow
    # Q: what is type of video_frames[i]? np.array? list?
    #    ... probably some imageio class
    next_step_pil = (
        Image.fromarray(video_frames.get_data(next_frame_n))
        .convert("RGB")
        .resize(img.image_shape, Image.LANCZOS)
    )

    # Apply flows
    flow_im = None
    for j, optical_flow in enumerate(optical_flows):
        # This looks like something that we shouldn't have to recompute
        # but rather could be attached to the flow object as an attribute
        old_frame_n = frame_n - (2**j - 1) * frame_stride
        save_n = i // save_every - (2**j - 1)
        if old_frame_n < 0 or save_n < 1:
            break

        current_step_pil = (
            Image.fromarray(video_frames.get_data(old_frame_n))
            .convert("RGB")
            .resize(img.image_shape, Image.LANCZOS)
        )

        filename = f"backup/{file_namespace}/{base_name}_{save_n}.bak"
        filename = None if j == 0 else filename
        # `flow_im` isn't being used for anything.
        # Might be interesting to log it for monitoring
        # Q: why does this function take `_step_pil` objects
        #    as well as a filename? `set_flow` automatically writes a
        #    backup file? That seems like something that should be
        #    configurable.
        flow_im, mask_tensor = optical_flow.set_flow(
            current_step_pil,
            next_step_pil,
            img,
            filename,
            infill_mode,
            sampling_mode,
            device=device,
        )

        optical_flow.set_enabled(True)
        # first flow is previous frame
        if j == 0:
            mask_accum = mask_tensor.detach()
            valid = mask_tensor.mean()
            logger.debug("valid pixels:", valid)
            # what is this magic number here?
            if reencode_each_frame or valid < 0.03:
                if isinstance(img, PixelImage) and valid >= 0.03:
                    img.lock_pallet()
                    img.encode_image(next_step_pil, smart_encode=False)
                    img.lock_pallet(lock_palette)
                else:
                    img.encode_image(next_step_pil)
                # this variable is unused...
                reencoded = True
            else:
                reencoded = False
        else:
            with torch.no_grad():
                optical_flow.set_mask((mask_tensor - mask_accum).clamp(0, 1))
                mask_accum.add_(mask_tensor)

    return flow_im, next_step_pil
