# Copyright (c) OpenMMLab. All rights reserved.
from operator import attrgetter
from typing import List, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from mmcv.cnn import ConvModule

if not hasattr(ConvModule, 'turn_on_fast_conv_bn_eval'):
    raise RuntimeError(
        'The MMCV you use does not support "fast_conv_bn_eval" feature.'
        'Please install the latest version of MMCV to use this feature.')


# Helper function to split a qualname into parent path and last atom.
def _parent_name(target: str) -> Tuple[str, str]:
    """Splits a qualname into parent path and last atom.

    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


def replace_sub_module(model, name, new_module):
    # Remove the original module from the model
    # usage: replace_sub_module(model, 'layer1.block2.conv2', conv)
    parent_name, name = _parent_name(name)
    if parent_name != '':
        getter = attrgetter(parent_name)
        parent = getter(model)
    else:
        parent = model
    setattr(parent, name, new_module)


def turn_on_fast_conv_bn_eval_for_single_model(model: torch.nn.Module):

    # first, turn on fast_conv_bn_eval feature for existing ConvModule
    for name, module in model.named_modules():
        if isinstance(module, ConvModule):
            module.turn_on_fast_conv_bn_eval()

    # second, merge consecutive conv+bn into ConvModule for the given model

    # Symbolically trace the input model to create an FX GraphModule
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

        # check if the conv and bn modules are used in multiple nodes
        conv_name = node.args[0].target
        bn_name = node.target

        conv_usage_count = 0
        bn_usage_count = 0
        for _node in fx_model.graph.nodes:
            if _node.op != 'call_module':
                continue
            if _node.target == conv_name:
                conv_usage_count += 1
            if _node.target == bn_name:
                bn_usage_count += 1

        if conv_usage_count > 1 or bn_usage_count > 1:
            continue

        # Find a pair of conv and bn to optimize
        conv_module = modules[conv_name]
        bn_module = modules[bn_name]

        # Fuse conv and bn into a ConvModule
        new_conv = ConvModule.create_from_conv_bn(conv_module, bn_module)
        replace_sub_module(model, conv_name, new_conv)
        replace_sub_module(model, bn_name, nn.Identity())


def turn_on_fast_conv_bn_eval(model: torch.nn.Module, modules: List[str]):
    for module_name in modules:
        module = attrgetter(module_name)(model)
        turn_on_fast_conv_bn_eval_for_single_model(module)
