# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from operator import attrgetter
from typing import List, Union

import torch
import torch.nn as nn


def efficient_conv_bn_eval_forward(bn: nn.modules.batchnorm._BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """Code borrowed from mmcv 2.0.1, so that this feature can be used for old
    mmcv versions.

    Implementation based on https://arxiv.org/abs/2305.11624
    "Tune-Mode ConvBN Blocks For Efficient Transfer Learning"
    It leverages the associative law between convolution and affine transform,
    i.e., normalize (weight conv feature) = (normalize weight) conv feature.
    It works for Eval mode of ConvBN blocks during validation, and can be used
    for training as well. It reduces memory and computation cost.
    Args:
        bn (_BatchNorm): a BatchNorm module.
        conv (nn._ConvNd): a conv module
        x (torch.Tensor): Input feature map.
    """
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var +
                               bn.eps).reshape([-1] + [1] *
                                               (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() *\
        (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


def bn_once_identity_forward(bn: nn.modules.batchnorm._BatchNorm,
                             x: torch.Tensor):
    """The forward function is an identity function.

    The magic is that after one call, the `bn.forward` will be restored to what
    it used to be.
    """
    bn.__dict__.pop('forward')
    return x


def efficient_conv_bn_eval_control(bn: nn.modules.batchnorm._BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    """This function controls whether to use `efficient_conv_bn_eval_forward`.

    If the following `bn` is in `eval` mode, then we turn on the special
    `efficient_conv_bn_eval_forward` and let the following call of `bn.forward`
    to be identity. Note that this `bn.forward` modification only works for one
    call. After the call, `bn.forward` will be restored to the default
    function. This is to deal with the case where one `bn` module is used in
    multiple places.
    """
    if not bn.training:
        # bn in eval mode
        output = efficient_conv_bn_eval_forward(bn, conv, x)
        bn.forward = partial(bn_once_identity_forward, bn)
        return output
    else:
        return conv._conv_forward(x, conv.weight, conv.bias)


def turn_on_efficient_conv_bn_eval_for_single_model(model: torch.nn.Module):
    # optimize consecutive conv+bn by modifying forward function
    # Symbolically trace the input model to create an FX GraphModule
    import torch.fx as fx
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())

    patterns = [(torch.nn.modules.conv._ConvNd,
                 torch.nn.modules.batchnorm._BatchNorm)]

    # Iterate through nodes in the graph to find ConvBN blocks
    for node in fx_model.graph.nodes:
        # If our current node isn't calling a Module then we can ignore it.
        if node.op != 'call_module':
            continue
        target_module = modules[node.target]
        found_pair = False
        for conv_class, bn_class in patterns:
            if isinstance(target_module, bn_class):
                source_module = modules[node.args[0].target]
                if isinstance(source_module, conv_class):
                    found_pair = True
        # Not a conv-BN pattern or output of conv is used by other nodes
        if not found_pair or len(node.args[0].users) > 1:
            continue

        # check if the conv modules are used in multiple nodes
        conv_name = node.args[0].target
        bn_name = node.target

        conv_usage_count = 0
        for _node in fx_model.graph.nodes:
            if _node.op != 'call_module':
                continue
            if _node.target == conv_name:
                conv_usage_count += 1

        if conv_usage_count > 1:
            continue

        # Find a pair of conv and bn to optimize
        conv_module = modules[conv_name]
        bn_module = modules[bn_name]

        conv_module.forward = partial(efficient_conv_bn_eval_control,
                                      bn_module, conv_module)


def turn_on_efficient_conv_bn_eval(model: torch.nn.Module,
                                   modules: Union[List[str], str]):
    if isinstance(modules, str):
        modules = [modules]
    for module_name in modules:
        module = attrgetter(module_name)(model)
        turn_on_efficient_conv_bn_eval_for_single_model(module)
