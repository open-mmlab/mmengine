# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from itertools import chain

import torch
import torch.nn as nn
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch._ops.ops.aten
tensor_like_constructor = (
    aten.empty_like.default,
    aten.empty_like.out,
    aten.full_like.default,
    aten.full_like.out,
    aten.ones_like.default,
    aten.ones_like.out,
    aten.rand_like.default,
    aten.rand_like.out,
    aten.randn_like.default,
    aten.randn_like.out,
    aten.randint_like.default,
    aten.randint_like.out,
    aten.randint_like.low_dtype,
    aten.randint_like.low_dtype_out,
    aten.zeros_like.default,
    aten.zeros_like.out,
    aten.new_empty.default,
    aten.new_empty.out,
    aten.new_empty_strided.default,
    aten.new_empty_strided.out,
    aten.new_full.default,
    aten.new_full.out,
    aten.new_zeros.default,
    aten.new_zeros.out,
    aten.new_ones.default,
    aten.new_ones.out,
)


def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes())


def _is_tensor_constructor(func: OpOverload):
    if func in tensor_like_constructor:
        return True
    assert isinstance(func, OpOverload)
    schema = func._schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return False
    # TODO: no real reason to restrict multiple outputs
    return (len(schema.returns) == 1
            and schema.returns[0].type is torch._C.TensorType.get())


class MetaTensorContext(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if _is_tensor_constructor(func):
            device_idx = [arg.name
                          for arg in func._schema.arguments].index('device')
            if len(args) > device_idx:
                args = list(args)
                args[device_idx] = 'meta'
            else:
                kwargs['device'] = 'meta'
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


def _load_state_dict_meta(model, state_dict):
    unexpected_keys = set(state_dict.keys())
    missing_keys = set()

    def load(module: nn.Module, prefix: str = ''):
        if type(module).load_state_dict != nn.Module.state_dict:
            warnings.warn(f'{type(module)}.load_state_dict will not take '
                          'effect since skip_init_weights=True')
        for name, tensor in chain(
                module.named_parameters(recurse=False),
                module.named_buffers(recurse=False)):
            tensor_name = f'{prefix}.{name}' if prefix else name
            if tensor_name not in state_dict:
                missing_keys.add(tensor_name)
            elif tensor.device.type != 'meta':
                # If MetaTensorContext does not initiate tensor with
                # meta device, just load the tensor inplace
                tensor.copy_(state_dict[tensor_name])
            else:
                module._parameters[tensor_name] = state_dict[tensor_name]
                unexpected_keys.remove(tensor_name)
        for name, module in module.named_children():
            load(module, f'{prefix}.{name}.')

    # TODO: Use torchdistX to accelerate loading checkpoint
    load(module=model)
    if unexpected_keys:
        warnings.warn(f'Unexpected key(s) {", ".join(unexpected_keys)} '
                      f'will not be loaded into model')
    if missing_keys:
        raise KeyError(
            f'Missing key(s) {", ".join(missing_keys)}. Each parameter'
            'and buffer must be loaded from checkpoint when '
            'skip_init_weights=True')
