import torch as th
import torch.nn as nn
from bovo.data.warp import get_identity_wrap

"""
Module that takes two black and white images I and J as input and predicts the deformation in the that alignes I to J
Simple convolution layers to a defined coarse resolution where the flow is predicted then bilinear upsampling to predict the final flow.
"""


class FlowPredictionModule(nn.Module):
    def __init__(self, img_height, img_width, img_depth, n_layers=3, first_depth=8):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.identity_wrap = nn.Parameter(
            get_identity_wrap(self.img_height, self.img_width), requires_grad=False
        )

        n_channels = first_depth
        self.encoding_layer = nn.Conv2d(
            self.img_depth * 2, n_channels, kernel_size=3, padding="same"
        )

        n_modules = 4  # TODO : add layers that do not downsample
        assert self.img_height % (2**n_modules) == 0
        assert self.img_width % (2**n_modules) == 0
        self.coarse_img_height = self.img_height // (2**n_modules)
        self.coarse_img_width = self.img_width // (2**n_modules)

        self.layers = []
        for i in range(n_modules):
            for _ in range(n_layers - 1):
                self.layers.append(ResNetBlock(n_channels, n_channels, False))
            self.layers.append(ResNetBlock(n_channels, n_channels * 2, True))
            n_channels *= 2
        self.layers = th.nn.ModuleList(self.layers)

        self.flow_channels = 2
        self.flow_layer = th.nn.Conv2d(n_channels, self.flow_channels, 1)

        self.upsampling = nn.UpsamplingBilinear2d(
            size=(self.img_height, self.img_width)
        )

    def forward(self, I: th.Tensor, J: th.Tensor):
        """
        I : batch, depth, height, width
        J : batch, depth, height, width
        """

        stack = th.concat((I, J), dim=1)
        x = self.encoding_layer(stack)

        for layer in self.layers:
            x = layer(x)

        coarse_flow = self.flow_layer(x)
        fine_flow = self.upsampling(coarse_flow)
        fine_grid = th.permute(fine_flow, (0, 2, 3, 1))
        return fine_grid

    def loss(self, I, J, W):
        W_pred = self(I, J)
        loss = th.mean(th.square((W - W_pred)))
        return loss

    def warp_losses(self, I, Ip, J, W):
        FIJ = self(I, J)
        FJIp = self(J, Ip)
        FIIp = self(I, Ip)

        phi = th.permute(
            nn.functional.grid_sample(
                th.permute(FJIp, (0, 3, 1, 2)),
                FIJ + self.identity_wrap,
                align_corners=True,
                padding_mode="border",
            ),
            (0, 2, 3, 1),
        )
        Lwvis = th.mean(th.square((FIJ + phi - W)))

        Lwarp = th.mean(th.square((FIIp - W)))

        return Lwvis, Lwarp


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
            # self.shortcut = nn.Sequential()
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)
