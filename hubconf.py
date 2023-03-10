import os
import sys

from model_segnet import SegNet as segnet
#sys.path.append(os.path.join(sys.path[0], 'PyTorch/Detection/SSD'))

# resnet18 is the name of entrypoint
# def segnet(**kwargs):
#     """ # This docstring shows up in hub.help()
#     SegNet model
#     """
#     # Call the model, load pretrained weights
#     model = _segnet(**kwargs)
#     return model
