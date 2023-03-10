dependencies = ['torch']
from model_segnet import SegNet as _segnet

# resnet18 is the name of entrypoint
def segnet(**kwargs):
    """ # This docstring shows up in hub.help()
    SegNet model
    """
    # Call the model, load pretrained weights
    model = _segnet(**kwargs)
    return model
