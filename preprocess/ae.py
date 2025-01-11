import torch.nn as nn
import torch

class Autoencoder3D(nn.Module):
    def __init__(self):
        
        super(Autoencoder3D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            SubDown(1,16),
            SubDown(16,32),
        )
        # Decoder
        self.decoder = nn.Sequential(
            SubUp(32,16,False),
            SubUp(16,1,True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SubDown(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv3d = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.mp3d = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.mp3d(x)
        return x
    
class SubUp(torch.nn.Module):
    def __init__(self, n_in, n_out, is_last):
        super().__init__()
        self.convt3d = nn.ConvTranspose3d(n_in, n_out, kernel_size=4, stride=2, padding=1)
        if is_last:
            self.last = nn.Sigmoid()
        else:
            self.last = nn.ReLU()
        
    def forward(self, x):
        x = self.convt3d(x)
        x = self.last(x)
        return x