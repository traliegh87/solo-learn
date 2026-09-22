from pretrainedmodels import xception as _pretrainedmodels_xception
from torch import nn


def xception(**kwargs):
    model = _pretrainedmodels_xception(pretrained=None, **kwargs)
    # drop the classification head -- BYOL needs a plain feature extractor
    model.last_linear = nn.Identity()
    # timm-style attribute solo-learn's base.py reads for non-resnet backbones
    model.num_features = 2048
    return model
