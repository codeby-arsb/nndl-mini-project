import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, up_channels, skip_channels):
        """
        in_channels: channels from previous layer
        up_channels: channels after ConvTranspose2d
        skip_channels: channels from the skip connection
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(up_channels + skip_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x_up = self.up(x)
        # Check shapes (height and width)
        if x_up.shape[2:] != skip.shape[2:]:
            raise ValueError(f"Shape mismatch in decoder! x_up: {x_up.shape}, skip: {skip.shape}")
        
        # Concatenate along channel dimension (dim=1)
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.relu(self.bn(x_cat))

class ColorizationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(1, 64)       # output: 64, 128x128
        self.enc2 = EncoderBlock(64, 128)     # output: 128, 64x64
        self.enc3 = EncoderBlock(128, 256)    # output: 256, 32x32
        self.enc4 = EncoderBlock(256, 512)    # output: 512, 16x16
        self.enc5 = EncoderBlock(512, 512)    # output: 512, 8x8
        
        # Bottleneck (deepest resolution 4x4)
        self.bottleneck = EncoderBlock(512, 512) # output: 512, 4x4
        
        # Decoder
        # D1: in=512, up=512, skip=512 => out=1024, 8x8
        self.dec1 = DecoderBlock(512, 512, 512)
        # D2: in=1024, up=512, skip=512 => out=1024, 16x16
        self.dec2 = DecoderBlock(1024, 512, 512)
        # D3: in=1024, up=256, skip=256 => out=512, 32x32
        self.dec3 = DecoderBlock(1024, 256, 256)
        # D4: in=512, up=128, skip=128 => out=256, 64x64
        self.dec4 = DecoderBlock(512, 128, 128)
        # D5: in=256, up=64, skip=64 => out=128, 128x128
        self.dec5 = DecoderBlock(256, 64, 64)
        
        # Final Output Layer
        # No skip connection here, just upsample to 256x256 and map to 2 channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x, return_shapes=False):
        shapes = {}
        if return_shapes: shapes['Input'] = list(x.shape)
        
        # Encoder
        e1 = self.enc1(x)
        if return_shapes: shapes['Encoder 1'] = list(e1.shape)
        
        e2 = self.enc2(e1)
        if return_shapes: shapes['Encoder 2'] = list(e2.shape)
        
        e3 = self.enc3(e2)
        if return_shapes: shapes['Encoder 3'] = list(e3.shape)
        
        e4 = self.enc4(e3)
        if return_shapes: shapes['Encoder 4'] = list(e4.shape)
        
        e5 = self.enc5(e4)
        if return_shapes: shapes['Encoder 5'] = list(e5.shape)
        
        # Bottleneck
        b = self.bottleneck(e5)
        if return_shapes: shapes['Bottleneck'] = list(b.shape)
        
        # Decoder
        d1 = self.dec1(b, e5)
        if return_shapes: 
            shapes['Decoder 1'] = list(d1.shape)
            shapes['Skip 1'] = {'enc': list(e5.shape), 'dec_before': list(self.dec1.up(b).shape), 'concat': list(d1.shape)}
        
        d2 = self.dec2(d1, e4)
        if return_shapes: 
            shapes['Decoder 2'] = list(d2.shape)
            shapes['Skip 2'] = {'enc': list(e4.shape), 'dec_before': list(self.dec2.up(d1).shape), 'concat': list(d2.shape)}
        
        d3 = self.dec3(d2, e3)
        if return_shapes: 
            shapes['Decoder 3'] = list(d3.shape)
            shapes['Skip 3'] = {'enc': list(e3.shape), 'dec_before': list(self.dec3.up(d2).shape), 'concat': list(d3.shape)}
        
        d4 = self.dec4(d3, e2)
        if return_shapes: 
            shapes['Decoder 4'] = list(d4.shape)
            shapes['Skip 4'] = {'enc': list(e2.shape), 'dec_before': list(self.dec4.up(d3).shape), 'concat': list(d4.shape)}
        
        d5 = self.dec5(d4, e1)
        if return_shapes: 
            shapes['Decoder 5'] = list(d5.shape)
            shapes['Skip 5'] = {'enc': list(e1.shape), 'dec_before': list(self.dec5.up(d4).shape), 'concat': list(d5.shape)}
        
        # Final
        out = self.final(d5)
        if return_shapes: shapes['Final Output'] = list(out.shape)
        
        if return_shapes:
            return out, shapes
            
        return out
