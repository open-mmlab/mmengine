from mmengine.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class CustomOptim:
    pass
