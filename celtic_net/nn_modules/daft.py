# Modified version of the code from https://github.com/ai-med/DAFT

import torch
from collections import OrderedDict

class DAFT(torch.nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_momentum: float = 0.1,
        stride: int = 1,
        ndim_non_img: int = 15,
        activation: str = "linear",
        embedding_factor: int = 7
    ) -> None:

        super().__init__()
        
        self.bn1 = torch.nn.BatchNorm3d(out_channels, momentum=bn_momentum, affine=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.global_pool = torch.nn.AdaptiveAvgPool3d(1)
        if stride != 1 or in_channels != out_channels:
            assert 'err'
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm3d(out_channels, momentum=bn_momentum),
            )
        else:
            self.downsample = None
            
        self.film_dims = 0
        self.film_dims = in_channels
 
        if activation == "sigmoid":
            self.scale_activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.scale_activation = torch.nn.Tanh()
        elif activation == "relu":
            self.scale_activation = torch.nn.ReLU()
        elif activation == "leaky_relu":
            self.scale_activation = torch.nn.LeakyReLU()    
        elif activation == "linear":
            self.scale_activation = None
        
        self.embedding_factor = embedding_factor
        aux_input_dims = self.film_dims
        
        # shift and scale = true
        self.split_size = self.film_dims
        self.scale = None
        self.shift = None
        self.film_dims = 2 * self.film_dims
        
        bottleneck_dim = int((ndim_non_img + aux_input_dims) // self.embedding_factor)

        assert bottleneck_dim > 0, 'bottleneck_dim is 0'
        print(f'bottleneck_dim={bottleneck_dim}')

        layers = [
            ("aux_base", torch.nn.Linear(ndim_non_img + aux_input_dims, bottleneck_dim, bias=False)),
            ("aux_relu", torch.nn.ReLU()),
            ("aux_out", torch.nn.Linear(bottleneck_dim, self.film_dims, bias=False)),
        ]

        self.aux = torch.nn.Sequential(OrderedDict(layers))
        
    def forward(self, feature_map, x_aux):
        
        out = self.rescale_features(feature_map, x_aux)
        return out
    
    def __str__(self):
        out_str = 'embedding factor: {:s} | activation: {:s}'.format(
            str(self.embedding_factor),
            str(self.scale_activation)
        )
        return out_str
    
    def rescale_features(self, feature_map, x_aux):
        
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)   
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
                    
        v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        if self.scale_activation is not None:
            v_scale = self.scale_activation(v_scale)
        
        return (v_scale * feature_map) + v_shift  
    