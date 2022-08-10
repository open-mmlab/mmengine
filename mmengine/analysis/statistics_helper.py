# Modified from
# https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/print_model_statistics.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import tabulate
import torch
from torch import nn


def _format_size(x: int, sig_figs: int = 3, hide_zero: bool = False) -> str:
    """Formats an integer for printing in a table or model representation.

    Expresses the number in terms of 'kilo', 'mega', etc., using
    'K', 'M', etc. as a suffix.
    Args:
        x (int) : The integer to format.
        sig_figs (int) : The number of significant figures to keep
        hide_zero (bool) : If True, x=0 is replaced with an empty string
            instead of '0'.
    Returns:
        str : The formatted string.
    """
    if hide_zero and x == 0:
        return ''

    def fmt(x: float) -> str:
        # use fixed point to avoid scientific notation
        return f'{{:.{sig_figs}f}}'.format(x).rstrip('0').rstrip('.')

    if abs(x) > 1e14:
        return fmt(x / 1e15) + 'P'
    if abs(x) > 1e11:
        return fmt(x / 1e12) + 'T'
    if abs(x) > 1e8:
        return fmt(x / 1e9) + 'G'
    if abs(x) > 1e5:
        return fmt(x / 1e6) + 'M'
    if abs(x) > 1e2:
        return fmt(x / 1e3) + 'K'
    return str(x)


def _pretty_statistics(statistics: Dict[str, Dict[str, int]],
                       sig_figs: int = 3,
                       hide_zero: bool = False) -> Dict[str, Dict[str, str]]:
    """Converts numeric statistics to strings with kilo/mega/giga/etc.

    labels.
    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types.
        sig_figs (int) : the number of significant figures for each stat
        hide_zero (bool) : if True, statistics that are zero will be
            written as an empty string. Defaults to False.
    Return:
        dict(str, dict(str, str)) : the input statistics as pretty strings
    """
    out_stats = {}
    for mod, stats in statistics.items():
        out_stats[mod] = {
            s: _format_size(val, sig_figs, hide_zero)
            for s, val in stats.items()
        }
    return out_stats


def _group_by_module(
        statistics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Converts statistics organized first by statistic type and then by module
    to statistics organized first by module and then by statistic type.

    Args:
        statistics (dict(str, dict(str, any))) : the statistics to convert
    Returns:
        dict(str, dict(str, any)) : the reorganized statistics
    """
    out_stats = defaultdict(dict)  # type: Dict[str, Dict[str, Any]]
    for stat_name, stat in statistics.items():
        for mod, val in stat.items():
            out_stats[mod][stat_name] = val
    return dict(out_stats)


def _indicate_uncalled_modules(
    statistics: Dict[str, Dict[str, str]],
    stat_name: str,
    uncalled_modules: Set[str],
    uncalled_indicator: str = 'N/A',
) -> Dict[str, Dict[str, str]]:
    """If a module is in the set of uncalled modules, replace its statistics
    with the specified indicator, instead of using the existing string.

    Assumes the statistic is already formatting in string form.
    Args:
        statistics (dict(str, dict(str, str))) : the statistics to
            format. Organized as a dictionary over modules, which are
            each a dictionary over statistic types. Expects statistics
            have already been converted to strings.
        stat_name (str) : the name of the statistic being modified
        uncalled_modules set(str) : a set of names of uncalled modules.
        indicator (str) : the string that will be used to indicate
            unused modules. Defaults to 'N/A'.
    Returns:
        dict(str, dict(str, str)) : the modified statistics
    """

    stats_out = {mod: stats.copy() for mod, stats in statistics.items()}
    for mod in uncalled_modules:
        if mod not in stats_out:
            stats_out[mod] = {}
        stats_out[mod][stat_name] = uncalled_indicator
    return stats_out


def _remove_zero_statistics(
    statistics: Dict[str, Dict[str, int]],
    force_keep: Optional[Set[str]] = None,
    require_trivial_children: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Any module that has zero for all available statistics is removed from
    the set of statistics.

    This can help declutter the reporting of statistics
    if many submodules have zero statistics. Assumes the statistics have
    a model hierarchy starting with a root that has name ''.
    Args:
        statistics (dict(str, dict(str, int))) : the statistics to
            remove zeros from. Organized as a dictionary over modules,
            which are each a dictionary over statistic types.
        force_keep (set(str) or None) : a set of modules to always keep, even
            if they are all zero.
        require_trivial_children (bool) : If True, a statistic will only
            be deleted if all its children are also deleted. Defaults to
            False.
    Returns:
        dict(str, dict(str, int)) : the input statistics dictionary,
            with submodules removed if they have zero for all statistics.
    """
    out_stats: Dict[str, Dict[str, int]] = {}
    _force_keep: Set[str] = force_keep if force_keep else set() | {''}

    def keep_stat(name: str) -> None:
        prefix = name + ('.' if name else '')
        trivial_children = True
        for mod in statistics:
            # 'if mod' excludes root = '', which is never a child
            if mod and mod.count('.') == prefix.count('.') and mod.startswith(
                    prefix):
                keep_stat(mod)
                trivial_children &= mod not in out_stats

        if ((not all(val == 0 for val in statistics[name].values()))
                or (name in _force_keep)
                or (require_trivial_children and not trivial_children)):
            out_stats[name] = statistics[name].copy()

    keep_stat('')
    return out_stats


def _fill_missing_statistics(
        model: nn.Module,
        statistics: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """If, for a given submodule name in the model, a statistic is missing from
    statistics, fills it in with zero.

    This visually uniformizes
    the reporting of statistics.
    Args:
        model (nn.Module) : the model whose submodule names will be
            used to fill statistics
        statistics (dict(str, dict(str, int))) : the statistics to
            fill in missing values for. Organized as a dictionary
            over statistics, which are each a dictionary over submodules'
            names. The statistics are assumed to be formatted already
            to the desired string format for printing.
    Returns:
        dict(str, dict(str, int)) : the input statistics with missing
            values filled with zero.
    """
    out_stats = {name: stat.copy() for name, stat in statistics.items()}
    for mod_name, _ in model.named_modules():
        for stat in out_stats.values():
            if mod_name not in stat:
                stat[mod_name] = 0
    return out_stats


def _model_stats_str(model: nn.Module,
                     statistics: Dict[str, Dict[str, str]]) -> str:
    """This produces a representation of the model much like 'str(model)'
    would, except the provided statistics are written out as additional
    information for each submodule.

    Args:
        model (nn.Module) : the model to form a representation of.
        statistics (dict(str, dict(str, str))) : the statistics to
            include in the model representations. Organized as a dictionary
            over module names, which are each a dictionary over statistics.
            The statistics are assumed to be formatted already to the
            desired string format for printing.
    Returns:
        str : the string representation of the model with the statistics
            inserted.
    """

    # Copied from nn.Module._addindent
    def _addindent(s_: str, numSpaces: int) -> str:
        s = s_.split('\n')
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def print_statistics(name: str) -> str:
        if name not in statistics:
            return ''
        printed_stats = [f'{k}: {v}' for k, v in statistics[name].items()]
        return ', '.join(printed_stats)

    # This comes directly from nn.Module.__repr__ with small changes
    # to include the statistics.
    def repr_with_statistics(module: nn.Module, name: str) -> str:
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = module.extra_repr()
        printed_stats = print_statistics(name)
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines.extend(extra_repr.split('\n'))
        if printed_stats:
            extra_lines.extend(printed_stats.split('\n'))
        child_lines = []
        for key, submod in module._modules.items():
            submod_name = name + ('.' if name else '') + key
            # pyre-fixme[6]: Expected `Module` for 1st param but got
            #  `Optional[nn.modules.module.Module]`.
            submod_str = repr_with_statistics(submod, submod_name)
            submod_str = _addindent(submod_str, 2)
            child_lines.append('(' + key + '): ' + submod_str)
        lines = extra_lines + child_lines

        main_str = module._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    return repr_with_statistics(model, '')


def _get_input_sizes(iterable: Iterable[Any]) -> List[Any]:  # pyre-ignore[2,3]
    """Gets the sizes of all torch tensors in an iterable.

    If an element of the iterable is a non-torch tensor iterable, it recurses
    into that iterable to continue calculating sizes. Any non-iterable is given
    a size of None. The output consists of nested lists with the same nesting
    structure as the input iterables.
    """
    out_list = []
    for i in iterable:
        if isinstance(i, torch.Tensor):
            out_list.append(list(i.size()))
        elif isinstance(i, Iterable):
            sublist_sizes = _get_input_sizes(i)
            if all(j is None for j in sublist_sizes):
                out_list.append(None)
            else:
                out_list.append(sublist_sizes)
        else:
            out_list.append(None)
    return out_list


def _get_single_child(name: str,
                      statistics: Dict[str, Dict[str, str]]) -> Optional[str]:
    """If the given module has only a single child in statistics, return it.

    Otherwise, return None.
    """
    prefix = name + ('.' if name else '')
    child = None
    for mod in statistics:
        # 'if mod' excludes root = '', which is never a child
        if mod and mod.count('.') == prefix.count('.') and mod.startswith(
                prefix):
            if child is None:
                child = mod
            else:
                return None  # We found a second child, so return None
    return child


def _try_combine(stats1: Dict[str, str],
                 stats2: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Try combine two statistics dict to display in one row.

    If they conflict, returns None.
    """
    ret = {}
    if set(stats1.keys()) != set(stats2.keys()):
        return None
    for k, v1 in stats1.items():
        v2 = stats2[k]
        if v1 != v2 and len(v1) and len(v2):
            return None
        ret[k] = v1 if len(v1) else v2
    return ret


def _fastforward(
        name: str,
        statistics: Dict[str, Dict[str, str]]) -> Tuple[str, Dict[str, str]]:
    """If the given module has only a single child and matches statistics with
    that child, merge statistics and their names into one row.

    Then repeat until the condition isn't met.
    Returns:
        str: the new name
        dict: the combined statistics of this row
    """
    single_child = _get_single_child(name, statistics)
    if single_child is None:
        return name, statistics[name]
    combined = _try_combine(statistics[name], statistics[single_child])
    if combined is None:
        return name, statistics[name]
    statistics[single_child] = combined
    return _fastforward(single_child, statistics)


def _model_stats_table(
    statistics: Dict[str, Dict[str, str]],
    max_depth: int = 3,
    stat_columns: Optional[List[str]] = None,
) -> str:
    """Formats the statistics obtained from a model in a nice table.

    Args:
        statistics (dict(str, dict(str, str))) : The statistics to print.
            Organized as a dictionary over modules, then as a dictionary
            over statistics in the model. The statistics are assumed to
            already be formatted for printing.
        max_depth (int) : The maximum submodule depth to recurse to.
        stat_columns (list(str)) : Specify the order of the columns to print.
            If None, columns are found automatically from the provided
            statistics.
    Return:
        str : The formatted table.
    """
    if stat_columns is None:
        stat_columns = set()
        for stats in statistics.values():
            stat_columns.update(stats.keys())
        stat_columns = list(stat_columns)

    headers = ['module'] + stat_columns
    table: List[List[str]] = []

    def build_row(name: str, stats: Dict[str, str],
                  indent_lvl: int) -> List[str]:
        indent = ' ' * indent_lvl
        row = [indent + name]
        for stat_name in stat_columns:  # Is not None at this point
            row_str = (indent + stats[stat_name]) if stat_name in stats else ''
            row.append(row_str)
        return row

    def fill(indent_lvl: int, prefix: str) -> None:
        if indent_lvl > max_depth:
            return
        for mod_name in statistics:
            # 'if mod' excludes root = '', which is never a child
            if (mod_name and mod_name.count('.') == prefix.count('.')
                    and mod_name.startswith(prefix)):
                mod_name, curr_stats = _fastforward(mod_name, statistics)
                if root_prefix and mod_name.startswith(root_prefix):
                    # Skip the root_prefix shared by all submodules as it
                    # carries 0 information
                    pretty_mod_name = mod_name[len(root_prefix):]
                else:
                    pretty_mod_name = mod_name
                row = build_row(pretty_mod_name, curr_stats, indent_lvl)
                table.append(row)
                fill(indent_lvl + 1, mod_name + '.')

    root_name, curr_stats = _fastforward('', statistics)
    row = build_row(root_name or 'model', curr_stats, indent_lvl=0)
    table.append(row)
    root_prefix = root_name + ('.' if root_name else '')
    fill(indent_lvl=1, prefix=root_prefix)

    old_ws = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    tab = tabulate.tabulate(table, headers=headers, tablefmt='pipe')
    tabulate.PRESERVE_WHITESPACE = old_ws
    return tab
