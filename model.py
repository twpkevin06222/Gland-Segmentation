import torch
from torch import nn
from torch import Tensor
from torchsummary import summary
import torch.nn.functional as F
# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 stride: tuple, kernel_size: tuple = (3, 3),
                 padding: str or int = 'same') -> None:
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ConvBlock(nn.Module):
    """
    Pre-activation convblock:
    batch norm -> activation -> weights
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: tuple = (1, 1), dropout: float = 0.1) -> None:
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = SeparableConv2d(in_channels, out_channels,
                                    stride=stride)
        self.drop = nn.Dropout2d(dropout, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.drop(x)

        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple = (3, 3)) -> None:
        super(UpBlock, self).__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              padding='same')

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.conv(x)

        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: tuple = (3, 3)) -> None:
        super(UpConvBlock, self).__init__()
        self.up = UpBlock(in_channels, out_channels, kernel_size)
        self.convblock1 = ConvBlock(out_channels*2, out_channels)
        self.convblock2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: Tensor, enc_feature: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([x, enc_feature], dim=1)
        x = self.convblock1(x)
        x = self.convblock2(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 downsampling: bool = True) -> None:
        super(ResBlock, self).__init__()
        self.downsampling = downsampling
        if self.downsampling:
            self.down_conv = SeparableConv2d(in_channels, out_channels,
                                             stride=(2, 2), padding='valid')
            self.convblock1 = ConvBlock(out_channels, out_channels)
        else:
            self.convblock1 = ConvBlock(in_channels, out_channels)
        self.convblock2 = ConvBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        if self.downsampling:
            # padding to maintain the power of 2 dimensionality
            x = F.pad(x, (1, 1, 1, 1), 'constant', 0)
            x = self.down_conv(x)
        identity = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x += identity

        return x


class Encoder(nn.Module):
    """
    Resnet inspired encoder
    """
    def __init__(self, in_channels: int, init_filter: int,
                 depth: int) -> None:
        super(Encoder, self).__init__()
        self.init_filter = init_filter
        self.in_channels = in_channels
        self.depth = depth
        self.conv = SeparableConv2d(in_channels, init_filter, stride=(1,1))
        self.resblock0 = ResBlock(init_filter, init_filter, downsampling=False)
        self.resblock = nn.ModuleList([ResBlock(self.init_filter*(2**i), self.init_filter*(2**(i+1)))
                                       for i in range(depth)])
        self.bn = nn.BatchNorm2d(init_filter*(2**(depth)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> list:
        block_outputs = []
        x = self.conv(x)
        x = self.resblock0(x)
        block_outputs.append(x)
        for i, block in enumerate(self.resblock):
            x = block(x)
            if i is self.depth-1:
                x = self.bn(x)
                x = self.relu(x)
            block_outputs.append(x)

        return block_outputs


class Decoder(nn.Module):
    def __init__(self, init_filter: int,
                 depth: int, n_class: int = 1) -> None:
        super(Decoder, self).__init__()
        self.init_filter = init_filter
        self.depth = depth
        self.conv = nn.Conv2d(init_filter, n_class, kernel_size=(3, 3),
                              stride=(1, 1), padding='same')
        self.upblock = nn.ModuleList([UpConvBlock(self.init_filter*(2**i), self.init_filter*(2**(i-1)))
                                     for i in range(depth, 0, -1)])
        self.bn1 = nn.BatchNorm2d(init_filter)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, enc_feature: list) -> Tensor:
        for i in range(len(self.upblock)):
            if i is 0:
                x = self.upblock[i](enc_feature[self.depth], enc_feature[self.depth-1])
            else:
                x = self.upblock[i](x, enc_feature[self.depth-(i+1)])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn2(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int,
                 init_filter: int, depth: int):
        super(UNet, self).__init__()
        self.enc = Encoder(in_channels, init_filter, depth)
        self.dec = Decoder(init_filter, depth)

    def forward(self, x):
        enc_feature = self.enc(x)
        output = self.dec(enc_feature)

        return output


if __name__ == '__main__':
    model = UNet(3, 64, 3).to(device)
    summary(model, (3, 512, 512))


