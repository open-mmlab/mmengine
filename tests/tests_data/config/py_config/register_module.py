from mmcv.cnn import CONV_LAYERS


@CONV_LAYERS.register_module()
class NewConv1:
    pass


@CONV_LAYERS.register_module()
class NewConv2:
    pass
