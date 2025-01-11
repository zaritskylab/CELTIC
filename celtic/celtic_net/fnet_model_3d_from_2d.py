# Adapted from https://github.com/AllenCellModeling/pytorch_fnet/tree/release_1

import celtic.celtic_net.fnet_model
import torch

class Model(celtic_net.fnet_model.Model):
    def predict(self, signal):
        """Generate predicted image.

        Takes in a 5D-tensor, which implies a 3D-image (batch, channel, z, y, x).
        This method expects self.net to be a 2D-model, so predictions are done 1 z-slice at a time.
        """
        assert len(signal.size()) == 5, 'expected 5d Tensor'

        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.eval()
        prediction = torch.zeros(signal.size())
        for idx_z in range(signal.size()[2]):
            signal_zslice = signal[:, :, idx_z, :, :]
            if self.gpu_ids[0] >= 0:
                signal_zslice = signal_zslice.cuda(self.gpu_ids[0])
            else:
                print('predicting on CPU')
            signal_zslice = torch.autograd.Variable(signal_zslice, volatile=True)
            prediction_zslice = module(signal_zslice).data.cpu()
            prediction[:, :, idx_z, :, :] = prediction_zslice
        return prediction
