# via https://github.com/multimodalart/majesty-diffusion/blob/main/latent.ipynb


import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TF


class MakeCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        WholeCrop=0,
        WC_Allowance=10,
        WC_Grey_P=0.2,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.WholeCrop = WholeCrop
        self.WC_Allowance = WC_Allowance
        self.WC_Grey_P = WC_Grey_P
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose(
            [
                # T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    # scale=(0.9,0.95),
                    fill=-1,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.RandomPerspective(p=1, interpolation = T.InterpolationMode.BILINEAR, fill=-1,distortion_scale=0.2),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ]
        )

    def forward(self, input):
        gray = transforms.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [input.shape[0], 3, self.cut_size, self.cut_size]
        output_shape_2 = [input.shape[0], 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
            ),
            **padargs
        )
        cutouts_list = []

        if self.Overview > 0:
            cutouts = []
            cutout = resize(pad_input, out_shape=output_shape, antialiasing=True)
            output_shape_all = list(output_shape)
            output_shape_all[0] = self.Overview * input.shape[0]
            pad_input = pad_input.repeat(input.shape[0], 1, 1, 1)
            cutout = resize(pad_input, out_shape=output_shape_all)
            if aug:
                cutout = self.augs(cutout)
            cutouts_list.append(cutout)

        if self.InnerCrop > 0:
            cutouts = []
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].add(1).div(2).clamp(0, 1).squeeze(0)).save(
                    "content/diff/cutouts/cutout_InnerCrop.jpg", quality=99
                )
            cutouts_tensor = torch.cat(cutouts)
            cutouts = []
            cutouts_list.append(cutouts_tensor)
        cutouts = torch.cat(cutouts_list)
        return cutouts
