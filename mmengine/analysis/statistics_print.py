# Modified from
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/print_model_statistics.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Optional, Tuple

from .activation_count import ActivationCountAnalysis
from .flop_count import FlopCountAnalysis
from .parameter_count import parameter_count
from .statistics_helper import (_fill_missing_statistics, _group_by_module,
                                _indicate_uncalled_modules, _model_stats_str,
                                _model_stats_table, _pretty_statistics,
                                _remove_zero_statistics)


def specific_stats_str(
        flops: FlopCountAnalysis,
        activations: Optional[ActivationCountAnalysis] = None) -> str:
    """
    Calculates the parameters and flops of the model with the given inputs
    and returns a string representation of the model that includes the
    parameters and flops of every submodule. The string is structured
    to be similar that given by str(model), though it is not guaranteed to
    be identical in form if the default string representation of a module has
    been overridden. If a module has zero parameters and flops, statistics
    will not be reported for succinctness.
    The trace can only register the scope of a module if it is called
    directly, which means flops (and activations) arising from explicit
    calls to .forward() or to other python functions of the module will
    not be attributed to that module. Modules that are never called will
    have 'N/A' listed for their flops; this means they are either unused
    or their statistics are missing for this reason. Any such flops are still
    counted towards the parent
    Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> class InnerNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(x))
    >>> class TestNet(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(10,10)
    ...         self.fc2 = nn.Linear(10,10)
    ...         self.inner = InnerNet()
    ...     def forward(self, x):
    ...         return self.fc1(self.fc2(self.inner(x)))
    >>> inputs = torch.randn((1,10))
    >>> print(flop_count_str(FlopCountAnalysis(model, inputs)))
    TestNet(
      #params: 0.44K, #flops: 0.4K
      (fc1): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
      )
      (fc2): Linear(
        in_features=10, out_features=10, bias=True
        #params: 0.11K, #flops: 100
      )
      (inner): InnerNet(
        #params: 0.22K, #flops: 0.2K
        (fc1): Linear(
          in_features=10, out_features=10, bias=True
          #params: 0.11K, #flops: 100
        )
        (fc2): Linear(
          in_features=10, out_features=10, bias=True
          #params: 0.11K, #flops: 100
        )
      )
    )
    Args:
        flops (FlopCountAnalysis): the flop counting object
        activations (bool) : If given, the activations of each layer will
            also be calculated and included in the representation.
    Returns:
        str:
            a string representation of the model with the number of
            parameters and flops included.
    """
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    model = flops._model
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings('none')
    stats = {'#params': params, '#flops': flops.by_module()}

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings('none')
        stats['#acts'] = activations.by_module()

    all_uncalled = flops.uncalled_modules() | (
        activations.uncalled_modules() if activations is not None else set())
    stats = _fill_missing_statistics(model, stats)
    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(stats, force_keep=all_uncalled)
    stats = _pretty_statistics(stats, sig_figs=2)  # type: ignore
    stats = _indicate_uncalled_modules(
        stats,  # type: ignore
        '#flops',  # type: ignore
        flops.uncalled_modules())  # type: ignore
    if activations is not None:
        stats = _indicate_uncalled_modules(
            stats,  # type: ignore
            '#acts',  # type: ignore
            activations.uncalled_modules())  # type: ignore

    model_string = ''
    if all_uncalled:
        model_string += (
            'N/A indicates a possibly missing statistic due to how '
            'the module was called. Missing values are still included '
            "in the parent's total.\n")
    model_string += _model_stats_str(model, stats)  # type: ignore
    return model_string


def specific_stats_table(
    flops: FlopCountAnalysis,
    max_depth: int = 3,
    activations: Optional[ActivationCountAnalysis] = None,
    show_param_shapes: bool = True,
) -> str:
    """
    Format the per-module parameters and flops of a model in a table.
    It looks like this:
    ::
        | model                            | #parameters or shape| #flops    |
        |:---------------------------------|:--------------------|:----------|
        | model                            | 34.6M               | 65.7G     |
        |  s1                              |  15.4K              |  4.32G    |
        |   s1.pathway0_stem               |   9.54K             |   1.23G   |
        |    s1.pathway0_stem.conv         |    9.41K            |    1.23G  |
        |    s1.pathway0_stem.bn           |    0.128K           |           |
        |   s1.pathway1_stem               |   5.9K              |   3.08G   |
        |    s1.pathway1_stem.conv         |    5.88K            |    3.08G  |
        |    s1.pathway1_stem.bn           |    16               |           |
        |  s1_fuse                         |  0.928K             |  29.4M    |
        |   s1_fuse.conv_f2s               |   0.896K            |   29.4M   |
        |    s1_fuse.conv_f2s.weight       |    (16, 8, 7, 1, 1) |           |
        |   s1_fuse.bn                     |   32                |           |
        |    s1_fuse.bn.weight             |    (16,)            |           |
        |    s1_fuse.bn.bias               |    (16,)            |           |
        |  s2                              |  0.226M             |  7.73G    |
        |   s2.pathway0_res0               |   80.1K             |   2.58G   |
        |    s2.pathway0_res0.branch1      |    20.5K            |    0.671G |
        |    s2.pathway0_res0.branch1_bn   |    0.512K           |           |
        |    s2.pathway0_res0.branch2      |    59.1K            |    1.91G  |
        |   s2.pathway0_res1.branch2       |   70.4K             |   2.28G   |
        |    s2.pathway0_res1.branch2.a    |    16.4K            |    0.537G |
        |    s2.pathway0_res1.branch2.a_bn |    0.128K           |           |
        |    s2.pathway0_res1.branch2.b    |    36.9K            |    1.21G  |
        |    s2.pathway0_res1.branch2.b_bn |    0.128K           |           |
        |    s2.pathway0_res1.branch2.c    |    16.4K            |    0.537G |
        |    s2.pathway0_res1.branch2.c_bn |    0.512K           |           |
        |   s2.pathway0_res2.branch2       |   70.4K             |   2.28G   |
        |    s2.pathway0_res2.branch2.a    |    16.4K            |    0.537G |
        |    s2.pathway0_res2.branch2.a_bn |    0.128K           |           |
        |    s2.pathway0_res2.branch2.b    |    36.9K            |    1.21G  |
        |    s2.pathway0_res2.branch2.b_bn |    0.128K           |           |
        |    s2.pathway0_res2.branch2.c    |    16.4K            |    0.537G |
        |    s2.pathway0_res2.branch2.c_bn |    0.512K           |           |
        |    ............................. |    ......           |    ...... |
    Args:
        flops (FlopCountAnalysis): the flop counting object
        max_depth (int) : The max depth of submodules to include in the
            table. Defaults to 3.
        activations (ActivationCountAnalysis or None) : If given, include
            activation counts as an additional column in the table.
        show_param_shapes (bool) : If true, shapes for parameters will be
            included in the table. Defaults to True.
    Returns:
        str : The formatted table.
    Examples:
    ::
        print(flop_count_table(FlopCountAnalysis(model, inputs)))
    """
    params_header = '#parameters' + (' or shape' if show_param_shapes else '')
    flops_header, acts_header = '#flops', '#activations'

    model = flops._model
    # cast to dict since pyre doesn't like the implicit defaultdict->dict
    params = dict(parameter_count(model))

    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    flops.tracer_warnings('none')

    stats = {params_header: params, flops_header: flops.by_module()}
    stat_columns = [params_header, flops_header]

    if activations is not None:
        activations.unsupported_ops_warnings(False)
        activations.uncalled_modules_warnings(False)
        activations.tracer_warnings('none')
        stats[acts_header] = activations.by_module()
        stat_columns += [acts_header]

    stats = _group_by_module(stats)
    stats = _remove_zero_statistics(
        stats,  # type: ignore
        require_trivial_children=True)  # type: ignore
    stats = _pretty_statistics(stats, hide_zero=False)  # type: ignore
    stats = _indicate_uncalled_modules(
        stats,  # type: ignore
        flops_header,  # type: ignore
        flops.uncalled_modules() & stats.keys(),  # type: ignore
        uncalled_indicator='',  # type: ignore
    )
    if activations:
        stats = _indicate_uncalled_modules(
            stats,  # type: ignore
            acts_header,  # type: ignore
            activations.uncalled_modules() & stats.keys(),  # type: ignore
            uncalled_indicator='',  # type: ignore
        )

    # Swap in shapes for parameters or delete shapes from dict
    param_shapes: Dict[str, Tuple[int, ...]] = {
        k: tuple(v.shape)
        for k, v in model.named_parameters()
    }
    to_delete = []
    for mod in stats:
        if mod in param_shapes:
            if show_param_shapes:
                stats[mod][params_header] = str(
                    param_shapes[mod])  # type: ignore
            else:
                to_delete.append(mod)
    for mod in to_delete:
        del stats[mod]

    return _model_stats_table(
        statistics=stats,  # type: ignore
        max_depth=max_depth,
        stat_columns=stat_columns,
    )
