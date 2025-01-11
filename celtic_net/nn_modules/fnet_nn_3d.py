# Modified version of the code from https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1

import celtic_net.nn_modules.fnet_nn_3d_params

class Net(celtic_net.nn_modules.fnet_nn_3d_params.Net):
    def __init__(self, context, signals_are_masked):
        self.context = context
        self.signals_are_masked = signals_are_masked
        in_channels = 1
        super().__init__(depth=4, mult_chan=32, in_channels=in_channels)

