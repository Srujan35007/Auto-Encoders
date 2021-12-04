import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    '''
    input_shape: -1, 3, 256, 256
    output_shape: -1, init_dims*64, 4, 4
    '''
    def __init__(self, img_channels=3):
        super(Encoder, self).__init__()
        init_dims = 4
        self.layers = nn.Sequential(
            nn.Conv2d(img_channels, init_dims*1, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # -1, init_dims*1, 256, 256
            self.EncoderBlock(init_dims*1, init_dims*2, 4, 2, 1),
            # -1, init_dims*2, 128, 128
            self.EncoderBlock(init_dims*2, init_dims*4, 4, 2, 1),
            # -1, init_dims*4, 64, 64
            self.EncoderBlock(init_dims*4, init_dims*8, 4, 2, 1),
            # -1, init_dims*8, 32, 32
            self.EncoderBlock(init_dims*8, init_dims*16, 4, 2, 1),
            # -1, init_dims*16, 16, 16
            self.EncoderBlock(init_dims*16, init_dims*32, 4, 2, 1),
            # -1, init_dims*32, 8, 8
            nn.Conv2d(init_dims*32, init_dims*64, 4, 2, 1),
            nn.Tanh(),
            # -1, init_dims*64, 4, 4
        )
        self._init_weights()

    def EncoderBlock(self, ins, outs, ksize, stride, pad):
        return nn.Sequential(
            nn.Conv2d(ins, outs, ksize, stride, pad),
            nn.BatchNorm2d(outs),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d)):
                nn.init.normal_(module.weight, 0, 0.2)
                nn.init.constant_(module.bias, 0)
        print(f"{self.__class__} model weights initialized")

from torchsummary import summary
enc = Encoder()
summary(enc, (3,256,256))
