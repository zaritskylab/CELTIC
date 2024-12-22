import torch
from celtic_net.nn_modules.daft import DAFT

class Net(torch.nn.Module):
    def __init__(self,
                 depth=4,
                 mult_chan=32,
                 in_channels=1,
                 out_channels=1
    ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels       

        if self.context:
            self.daft_params = {'ndim_non_img': self.context['context_features_len'],
                                'activation': self.context['daft_scale_activation'],
                                'embedding_factor': self.context['daft_embedding_factor']
                                }
            
        else:
            self.daft_params = None

        self.net_recurse = _Net_recurse(n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth=self.depth, net_daft_params=self.daft_params)        
        self.conv_out = torch.nn.Conv3d(self.mult_chan*self.in_channels, self.out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, x_tabular=None):
        x_rec = self.net_recurse(x, x_tabular)             
        res = self.conv_out(x_rec) 
        return res
       
class _Net_recurse(torch.nn.Module):

    def __init__(self, n_in_channels, mult_chan=2, depth=0, net_daft_params=None):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        
        if net_daft_params and depth==0:
            conv_more_daft_params = net_daft_params
        else:
            conv_more_daft_params = None
        
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels, daft_params=conv_more_daft_params)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels, daft_params=None)  

            self.conv_down = torch.nn.Conv3d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose3d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1), net_daft_params=net_daft_params)

    def forward(self, x, x_tabular=None):
                   
        if self.depth == 0:
            
            x_2conv_more = self.sub_2conv_more(x, x_tabular)
            return x_2conv_more
        
        else: 
            
            # convolve twice
            x_2conv_more = self.sub_2conv_more(x, x_tabular)

            # downsample
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            
            # run the depth-1 u-net (recursively)
            x_sub_u = self.sub_u(x_relu0, x_tabular)
            
            # upsample
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
                      
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            
            x_2conv_less = self.sub_2conv_less(x_cat, x_tabular)
     
            return x_2conv_less

class SubNet2Conv(torch.nn.Module):

    def __init__(self, n_in, n_out, daft_params):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

        if daft_params:
            self.daft = DAFT(n_out, 
                            n_out, 
                            bn_momentum = 0.1, 
                            stride = 1, 
                            ndim_non_img = daft_params['ndim_non_img'], 
                            activation = daft_params['activation'], 
                            embedding_factor = daft_params['embedding_factor']
                            )   
            
            print(self.daft)

    def forward(self, x, x_tabular=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        if hasattr(self, 'daft'):
            x = self.daft(x, x_tabular)

        return x
