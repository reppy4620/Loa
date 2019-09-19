import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_nc, out_nc, ks, s, p, leaky=False):
        super(Down, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, ks, s, p),
            nn.BatchNorm2d(out_nc),
            nn.LeakyReLU(0.2) if leaky else nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


def InConv(img_nc, out_nc, leaky=False):
    return Down(img_nc, out_nc, 3, 1, 1, leaky)


def Down3x3(in_nc, out_nc, leaky=False):
    return Down(in_nc, out_nc, 3, 1, 1, leaky)


def Down4x4(in_nc, out_nc, leaky=False):
    return Down(in_nc, out_nc, 4, 2, 1, leaky)


class Up(nn.Module):
    def __init__(self, nc, ks1=4, ks2=3,
                 s1=2, s2=1, p1=1, p2=1):
        super(Up, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nc, int(nc / 2), ks1, s1, p1),
            nn.BatchNorm2d(nc // 2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(int(nc/2), int(nc/4), ks2, s2, p2),
            nn.BatchNorm2d(nc // 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # if x1.size is not equal to x2.size, padding small image.
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x//2, diff_x - diff_x//2,
                        diff_y//2, diff_y - diff_y//2])
        x = self.layer1(torch.cat([x2, x1], dim=1))
        return self.layer2(x)


class Out(nn.Module):

    def __init__(self, in_nc, out_nc, ks=3, s=1, p=1):
        super(Out, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, ks, s, p),
            nn.BatchNorm2d(out_nc),
            nn.Tanh()
        )

    def forward(self, x1, x2):
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.down(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=3, ngf=32):
        super(UNet, self).__init__()
        self.in_layer = InConv(in_channel, ngf)
        self.down1 = Down4x4(ngf, ngf*2)
        self.down2 = Down3x3(ngf*2, ngf*2)
        self.down3 = Down4x4(ngf*2, ngf*4)
        self.down4 = Down3x3(ngf*4, ngf*4)
        self.down5 = Down4x4(ngf*4, ngf*8)
        self.down6 = Down3x3(ngf*8, ngf*8)
        self.down7 = Down4x4(ngf*8, ngf*16)
        self.down8 = Down3x3(ngf*16, ngf*16)
        self.up1 = Up(ngf*32)
        self.up2 = Up(ngf*16)
        self.up3 = Up(ngf*8)
        self.up4 = Up(ngf*4)
        self.out = Out(ngf*2, out_channel)

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        x9 = self.down8(x8)

        x = self.up1(x8, x9)
        x = self.up2(x, x7)
        x = self.up3(x, x5)
        x = self.up4(x, x3)
        x = self.out(x, x1)
        return x
