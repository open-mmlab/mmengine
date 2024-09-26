# Frequently asked questions

- How to enable EpochBased/IterBased training?

  See this [tutorial](../common_usage/epoch_to_iter.md) for more details

- Accuracy discrepancy between validation and testing

  Try to enable `broadcast_buffer=True` in model_wrapper_cfg

  ```python
  model_wrapper_cfg = dict(
      type='MMDistributedDataParallel',
      broadcast_buffer=True,
  )
  ```
