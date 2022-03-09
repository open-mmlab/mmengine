from typing import Dict


def get_metric_value(indicator: str, metrics: Dict):
    """Get the metric value specified by a indicator, which can be the metric
    name or full name with evaluator prefix.

    Args:
        indicator (str): The metric indicator, which can be the metric name
            (e.g. 'AP') or the full name with prefix (e.g. 'COCO/AP')
        metrics (dict): The evaluation results output by the evaluator

    Returns:
        Any: The value of the specified metric
    """

    if '/' in indicator:
        if indicator in metrics:
            return metrics[indicator]
        else:
            raise ValueError(
                f'The indicator "{indicator}" can not match any metric in '
                f'{list(metrics.keys())}')
    else:
        matched = [k for k in metrics.keys() if k.split('/')[-1] == indicator]

        if not matched:
            raise ValueError(
                f'The indicator {indicator} can not match any metric in '
                f'{list(metrics.keys())}')
        elif len(matched) > 1:
            raise ValueError(f'The indicator "{indicator}" matches multiple '
                             f'metrics {matched}')
        else:
            return metrics[matched[0]]
