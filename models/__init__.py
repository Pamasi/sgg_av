from .reltr import build
from .reltr_log import build_log

def build_model(args, use_log=False):
    if not use_log:
        return build(args)
    else:
        return build_log(args)
