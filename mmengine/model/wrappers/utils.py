# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import MODEL_WRAPPERS


def is_model_wrapper(model):
    """Check if a module is a model wrapper.

    The following 4 model in MMEngine (and their subclasses) are regarded as
    model wrappers: DataParallel, DistributedDataParallel,
    MMDataParallel, MMDistributedDataParallel. You may add you own
    model wrapper by registering it to ``mmengine.registry.MODEL_WRAPPERS``.

    Args:
        model (nn.Module): The model to be checked.

    Returns:
        bool: True if the input model is a model wrapper.
    """
    model_wrappers = tuple(MODEL_WRAPPERS.module_dict.values())
    return isinstance(model, model_wrappers)
