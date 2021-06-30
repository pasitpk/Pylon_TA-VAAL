import torch
import torch.nn.functional as F
from torch import nn

from segmentation_models_pytorch.base import (SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.pan.decoder import ConvBnRelu


class Pylon(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 pretrain='imagenet',
                 n_dec_ch=128,
                 n_in=1,
                 n_out=14):
        super(Pylon, self).__init__()
        self.net = PylonCore(
            encoder_name=backbone,
            encoder_weights=pretrain,
            decoder_channels=n_dec_ch,
            in_channels=n_in,
            classes=n_out,
            upsampling=1,
            align_corners=True,
        )
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.net(x).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

        return {
            'pred': pred,
            'seg': seg,
        }


class PylonCore(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "resnet50",
            encoder_weights: str = "imagenet",
            decoder_channels: int = 128,
            in_channels: int = 1,
            classes: int = 1,
            upsampling: int = 1,
            align_corners=True,
    ):
        super(PylonCore, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        self.decoder = PylonDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            upscale_mode='bilinear',
            align_corners=align_corners,
        )

        self.segmentation_head = SegmentationHead(in_channels=decoder_channels,
                                                  out_channels=classes,
                                                  activation=None,
                                                  kernel_size=1,
                                                  upsampling=upsampling)

        # just to comply with SegmentationModel
        self.classification_head = None

        self.name = "pylon-{}".format(encoder_name)
        self.initialize()


class PylonDecoder(nn.Module):
    """returns each layer of decoder
    """
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            upscale_mode: str = 'bilinear',
            align_corners=True,
    ):
        super(PylonDecoder, self).__init__()

        self.pa = PA(
            in_channels=encoder_channels[-1],
            out_channels=decoder_channels,
            align_corners=align_corners,
        )

        kwargs = dict(
            out_channels=decoder_channels,
            upscale_mode=upscale_mode,
            align_corners=align_corners,
        )
        self.up3 = UP(
            in_channels=encoder_channels[-2],
            **kwargs,
        )
        self.up2 = UP(
            in_channels=encoder_channels[-3],
            **kwargs,
        )
        self.up1 = UP(
            in_channels=encoder_channels[-4],
            **kwargs,
        )

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.pa(bottleneck)  # 1/32
        x4 = self.up3(features[-2], x5)  # 1/16
        x3 = self.up2(features[-3], x4)  # 1/8
        x2 = self.up1(features[-4], x3)  # 1/4
        return x2


class PA(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            upscale_mode='bilinear',
            align_corners=True,
    ):
        super(PA, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None

        # middle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ))

        # pyramid attention branch
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=in_channels,
                       out_channels=1,
                       kernel_size=7,
                       stride=1,
                       padding=3))
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1,
                       out_channels=1,
                       kernel_size=5,
                       stride=1,
                       padding=2))
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       stride=1,
                       padding=1))

        self.conv3 = ConvBnRelu(in_channels=1,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv2 = ConvBnRelu(in_channels=1,
                                out_channels=1,
                                kernel_size=5,
                                stride=1,
                                padding=2)
        self.conv1 = ConvBnRelu(in_channels=1,
                                out_channels=1,
                                kernel_size=7,
                                stride=1,
                                padding=3)

    def forward(self, x):
        upscale_parameters = dict(mode=self.upscale_mode,
                                  align_corners=self.align_corners)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = F.interpolate(self.conv3(x3), scale_factor=2, **upscale_parameters)
        x = F.interpolate(self.conv2(x2) + x,
                          scale_factor=2,
                          **upscale_parameters)
        x = F.interpolate(self.conv1(x1) + x,
                          scale_factor=2,
                          **upscale_parameters)
        x = torch.mul(x, mid)
        return x


class UP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            upscale_mode: str = 'bilinear',
            align_corners=True,
    ):
        super(UP, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners if upscale_mode == 'bilinear' else None

        self.conv1 = ConvBnRelu(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0)

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y,
                             size=(h, w),
                             mode=self.upscale_mode,
                             align_corners=self.align_corners)
        conv = self.conv1(x)
        return y_up + conv


class PylonTA(nn.Module):
    def __init__(self,
                 backbone='resnet50',
                 pretrain='imagenet',
                 n_dec_ch=128,
                 n_in=1,
                 n_out=14,
                 loss_pred_hidden_size=128,
                 detach=True,):
        super(PylonTA, self).__init__()
        
        self.encoder = get_encoder(
            backbone,
            in_channels=n_in,
            depth=5,
            weights=pretrain,
        )

        self.decoder = PylonTADecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=n_dec_ch,
            upscale_mode='bilinear',
            align_corners=True,
        )

        self.segmentation_head = SegmentationHead(in_channels=n_dec_ch,
                                                  out_channels=n_out,
                                                  activation=None,
                                                  kernel_size=1,
                                                  upsampling=1)

        self.pool = nn.AdaptiveMaxPool2d(1)

        self.loss_predictor = LossPredictor(list(self.encoder.out_channels) + [n_dec_ch] * 4, loss_pred_hidden_size, detach)

    def forward(self, x):

        encoder_features = list(self.encoder(x))
        decoder_features = list(self.decoder(*encoder_features))
        # enforce float32 is a good idea
        # because if the loss function involves a reduction operation
        # it would be harmful, this prevents the problem
        seg = self.segmentation_head(decoder_features[0]).float()
        pred = self.pool(seg)
        pred = torch.flatten(pred, start_dim=1)

        features = encoder_features + decoder_features
        loss_pred = self.loss_predictor(features)

        return {
            'pred': pred,
            'seg': seg,
            'loss_pred': loss_pred,
        }


class PylonTADecoder(PylonDecoder):
    """return 
    """
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            upscale_mode: str = 'bilinear',
            align_corners=True,
    ):
        super(PylonTADecoder, self).__init__(encoder_channels, decoder_channels, upscale_mode, align_corners)

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.pa(bottleneck)  # 1/32
        x4 = self.up3(features[-2], x5)  # 1/16
        x3 = self.up2(features[-3], x4)  # 1/8
        x2 = self.up1(features[-4], x3)  # 1/4

        return x2, x3, x4, x5


class LossPredictor(nn.Module):
    def __init__(
            self,
            feature_channels,
            hidden_size=128,
            detach=True,
        ):
        super(LossPredictor, self).__init__()

        self.feature_blocks = self.get_feature_blocks(feature_channels, hidden_size)
        self.loss_predictor = nn.Linear(hidden_size*len(feature_channels), 1)
        self.detach = detach

    def forward(self, features):
        if self.detach:
            features = [f.detach() for f in features]
        
        pooled_features = torch.cat([fb(f) for fb, f in zip(self.feature_blocks, features)], 1)
        loss = self.loss_predictor(pooled_features)

        return loss

    def get_feature_blocks(self, feature_channels, hidden_size):
        feature_blocks = []
        for fc in feature_channels:
            feature_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(fc, hidden_size),
                nn.ReLU(),
            )
            feature_blocks.append(feature_block)
        
        feature_blocks = nn.ModuleList(feature_blocks)

        return feature_blocks

