from model_segnet import SegNet as segnet
dependencies = ['torch']

def testmodel():
    model = segnet()
    return model
