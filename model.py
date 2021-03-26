import torch
import torch.nn as nn
import torch.nn.functional as F




class CARAFE(nn.Module):
    def __init__(self, inC, outC, Kencoder=3, delta=2, Kup=5, Cm=64): # Kup = Kencoder + 2
        super(CARAFE, self).__init__()
        self.Kencoder = Kencoder
        self.delta = delta
        self.Kup = Kup
        self.down = nn.Conv2d(in_channels=inC, out_channels=Cm, kernel_size=1)  #
        self.encoder = nn.Conv2d(64, self.delta ** 2 * self.Kup ** 2,
                                 self.Kencoder, 1, self.Kencoder// 2)
        self.out = nn.Conv2d(inC, outC, 1)
 
    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()
 

        kernel_tensor = self.down(in_tensor)  
        kernel_tensor = self.encoder(kernel_tensor)  
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.delta)  
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  
        kernel_tensor = kernel_tensor.unfold(2, self.delta, step=self.delta) 
        kernel_tensor = kernel_tensor.unfold(3, self.delta, step=self.delta) 
        kernel_tensor = kernel_tensor.reshape(N, self.Kup ** 2, H, W, self.delta ** 2) 
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  
 
        in_tensor = F.pad(in_tensor, pad=(self.Kup // 2, self.Kup // 2,
                                          self.Kup // 2, self.Kup // 2),
                          mode='constant', value=0) 
        in_tensor = in_tensor.unfold(dimension=2, size=self.Kup, step=1) 
        in_tensor = in_tensor.unfold(3, self.Kup, step=1) 
        in_tensor = in_tensor.reshape(N, C, H, W, -1) 
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  
 
        out_tensor = torch.matmul(in_tensor, kernel_tensor)  
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.delta) 
        out_tensor = self.out(out_tensor)
        return out_tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = False
        if in_channels != out_channels:
            self.downsample = True

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        if self.downsample:
            skip = self.conv1x1(x)
        else:
            skip = x
        x_inc = self.inc(x)
        out = F.relu(skip + x_inc)
        return out



class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block,self).__init__()
        self.bconv = BasicConv2d( in_channels,out_channels,kernel_size=3,padding = 1)
        self.resnet = ResBlock(out_channels,out_channels)

    def forward(self,x):
        x1 = self.bconv(x)
        x2 = self.resnet(x1)
        return x2


class CARes_Unet(nn.Module):
    def __init__(self):
        super(CARes_Unet,self).__init__()
        self.encoder1 = Conv_Block(1,64)
        self.down1 = nn.Sequential(
            BasicConv2d(64, 64, 3, 2, 1),
            ResBlock(64, 64)
        )
        
        self.encoder2 = Conv_Block(64,128)
        self.down2 = nn.Sequential(
            BasicConv2d(128, 128, 3, 2, 1),
            ResBlock(128, 128)
        )
        
        self.encoder3 = Conv_Block(128,256)
        self.down3 = nn.Sequential(
            BasicConv2d(256, 256, 3, 2, 1),
            ResBlock(256, 256)
        )
        
        self.encoder4 = Conv_Block(256,512)
        self.down4 = nn.Sequential(
            BasicConv2d(512, 512, 3, 2, 1),
            ResBlock(512, 512)
        )
        
        self.decoder5 = Conv_Block(512,1024)
        self.up1 = CARAFE(1024, 512)
        
        self.decoder6 = Conv_Block(1024,512)
        self.up2 = CARAFE(512, 256)
        
        self.decoder7 = Conv_Block(512,256)
        self.up3 = CARAFE(256, 128)
        
        self.decoder8 = Conv_Block(256,128)
        self.up4 = CARAFE(128, 64)
        
        self.conv = Conv_Block(128,64)
        self.classfier = nn.Conv2d(64, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.encoder4(self.down3(x3))

        x5 = self.decoder5(self.down4(x4))
        output = torch.cat([x4,self.up1(x5)],dim = 1)
        output = self.decoder6(output)
        output = torch.cat([x3,self.up2(output)],dim=1)
        output = self.decoder7(output)
        output = torch.cat([x2,self.up3(output)],dim=1)
        output = self.decoder8(output)
        output = torch.cat([x1,self.up4(output)],dim=1)

        output = self.conv(output)
        output = self.classfier(output)
        output = self.sigmoid(output)
        return output