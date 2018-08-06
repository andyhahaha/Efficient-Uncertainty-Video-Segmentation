import torchvision.models as models

from ptsemseg.models.segnet import *
from ptsemseg.models.bayesian_segnet import *
from ptsemseg.models.tiramisu import *
def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == 'bayesian_segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == 'bayesian_tiramisu':
        model = model(n_classes=n_classes)
    else:
        raise 'Model {} not available'.format(name)

    return model

def _get_model_instance(name):
    return {
        'segnet': segnet,
        'bayesian_segnet': bayesian_segnet,
        'bayesian_tiramisu': FCDenseNet103
    }[name]
