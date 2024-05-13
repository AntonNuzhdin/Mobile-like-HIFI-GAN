import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1

# ORIGINAL
# class ResBlock1(torch.nn.Module):
#     def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
#         super(ResBlock1, self).__init__()
#         self.h = h
#         self.convs1 = nn.ModuleList([
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
#                                padding=get_padding(kernel_size, dilation[0]))),
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
#                                padding=get_padding(kernel_size, dilation[1]))),
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
#                                padding=get_padding(kernel_size, dilation[2])))
#         ])
#         self.convs1.apply(init_weights)

#         self.convs2 = nn.ModuleList([
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
#                                padding=get_padding(kernel_size, 1))),
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
#                                padding=get_padding(kernel_size, 1))),
#             weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
#                                padding=get_padding(kernel_size, 1)))
#         ])
#         self.convs2.apply(init_weights)

#     def forward(self, x):
#         for c1, c2 in zip(self.convs1, self.convs2):
#             xt = F.leaky_relu(x, LRELU_SLOPE)
#             xt = c1(xt)
#             xt = F.leaky_relu(xt, LRELU_SLOPE)
#             xt = c2(xt)
#             x = xt + x
#         return x

#     def remove_weight_norm(self):
#         for l in self.convs1:
#             remove_weight_norm(l)
#         for l in self.convs2:
#             remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[2]), dilation=dilation[2])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, 1), dilation=1)),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[0]), dilation=dilation[0])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            ),
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, groups=channels,
                                      padding=get_padding(kernel_size, dilation[1]), dilation=dilation[1])),
                weight_norm(nn.Conv1d(channels, channels, 1, 1))
            )
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# --- HIFI++ DISCRIMINATOR -- #

class DiscriminatorSPlusPlus(torch.nn.Module):
    def __init__(
        self, scale, factor=1.0, use_spectral_norm=False, mel_cond=False
    ):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.mel_cond = mel_cond
        self.activations = None
        self.pooling = (
            nn.Identity()
            if scale == 1
            else nn.AvgPool1d(scale, scale, padding=0)
        )
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, int(64 * factor), 15, 1, padding=7)),
                norm_f(
                    nn.Conv1d(
                        int(64 * factor),
                        int(64 * factor),
                        41,
                        2,
                        groups=4,
                        padding=20,
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        int(64 * factor),
                        int(128 * factor),
                        41,
                        2,
                        groups=16,
                        padding=20,
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        int(128 * factor),
                        int(256 * factor),
                        41,
                        4,
                        groups=16,
                        padding=20,
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        int(256 * factor),
                        int(512 * factor),
                        41,
                        4,
                        groups=16,
                        padding=20,
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        int(512 * factor),
                        int(512 * factor),
                        41,
                        1,
                        groups=16,
                        padding=20,
                    )
                ),
                norm_f(
                    nn.Conv1d(
                        int(512 * factor), int(512 * factor), 5, 1, padding=2
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv1d(int(512 * factor), 1, 3, 1, padding=1)
        )
        if self.mel_cond:
            self.embed = nn_utils.MultiScaleResnet(
                [513, 160, 320],
                4,
                mode="waveunet_k5_bn",
                out_width=80,
                in_width=513,
                scale_factor=2,
            )
            self.proj = norm_f(
                torch.nn.Conv1d(int(512 * factor), 80, 1, 1, padding=0)
            )

    def forward(self, x, mel=None):
        self.activations = []

        x = self.pooling(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            self.activations.append(x)
        h = x
        x = self.conv_post(x)
        self.activations.append(x)
        if self.mel_cond:
            h = self.proj(h)
            mel = mel.squeeze(1)
            emb = torch.nn.functional.interpolate(
                self.embed(mel), size=h.shape[-1], mode="linear"
            )
            x = x + torch.sum(emb * h, 1, keepdim=True)
        x = torch.flatten(x, 1, -1)

        return x

class MSDPlusPlus(torch.nn.Module):
    def __init__(
        self,
        factor=1.0,
        num_discs=1,
        use_spectral_norm=True,
    ):
        super().__init__()
        self.discs = nn.ModuleDict()
        for i in range(num_discs):
            self.discs[f"k{i+1}"] = DiscriminatorSPlusPlus(
                1,
                factor=factor,
                use_spectral_norm=use_spectral_norm,
            )

    def forward(self, x, mel=None):
        logits, activations = dict(), dict()
        for name, disc in self.discs.items():
            logits[name] = disc(x, mel=mel)
            if hasattr(disc, "activations"):
                activations[name] = disc.activations
        return logits, activations

# ---------------------------- #

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def feature_loss_plusplus(act1, act2):
    loss = 0.0
    for a1, a2 in zip(act1.values(), act2.values()):
        for rl, gl in zip(a1, a2):
            loss += torch.mean(torch.abs(rl - gl))
        # loss += F.l1_loss(torch.tensor(a1), torch.tensor(a2)).mean()
    return loss


def discriminator_loss_plusplus(logits_real, logits_fake):
    loss = 0
    for dr, dg in zip(logits_real.values(), logits_fake.values()):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs.values():
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
