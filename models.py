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


""" 
==========================================
Start of PylonTA
==========================================
"""

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

""" 
==========================================
End of PylonTA
==========================================
"""

""" 
==========================================
Start of BetaVAE
https://github.com/sinhasam/vaal/blob/master/model.py
==========================================
"""

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN.
        'input_size' must be even and larger than 16
    """
    def __init__(self, input_size=256, z_dim=32, r_dim=1, nc=3):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.output_size = input_size // 128
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024*self.output_size*self.output_size)),
        )

        self.fc_mu = nn.Linear(1024*self.output_size*self.output_size, z_dim) 
        self.fc_logvar = nn.Linear(1024*self.output_size*self.output_size, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + r_dim, 1024*self.output_size*2*self.output_size*2),
            View((-1, 1024, self.output_size*2, self.output_size*2)), 
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, nc, 1),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, r):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z_r = torch.cat([z, r], dim=1)
        x_recon = self._decode(z_r)

        return x_recon, z_r, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z_r):
        return self.decoder(z_r)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_r_dim=33):
        super(Discriminator, self).__init__()
        self.z_r_dim = z_r_dim
        self.net = nn.Sequential(
            nn.Linear(z_r_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z_r):
        return self.net(z_r)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

""" 
==========================================
End of BetaVAE
==========================================
"""