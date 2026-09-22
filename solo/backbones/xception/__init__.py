from .xception import xception as default_xception


def xception(method, *args, **kwargs):
    return default_xception(*args, **kwargs)


__all__ = ["xception"]
