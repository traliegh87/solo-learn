from pretrainedmodels import xception as _pretrainedmodels_xception
from pretrainedmodels.models.xception import Block
from torch import nn


def _init_weights(model: nn.Module):
    # Cadene's port ships with its init loop commented out, leaving PyTorch's default
    # kaiming_uniform(a=sqrt(5)); use the same init as timm's legacy_xception instead.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # zero-init the last BN of each residual branch so every Block starts as identity,
    # which keeps activations from compounding across the 8 identity-skip middle blocks
    for m in model.modules():
        if isinstance(m, Block):
            last_bn = [layer for layer in m.rep if isinstance(layer, nn.BatchNorm2d)][-1]
            nn.init.zeros_(last_bn.weight)


def xception(**kwargs):
    model = _pretrainedmodels_xception(pretrained=None, **kwargs)
    _init_weights(model)
    # drop the classification head -- BYOL needs a plain feature extractor
    model.last_linear = nn.Identity()
    # timm-style attribute solo-learn's base.py reads for non-resnet backbones
    model.num_features = 2048
    return model
