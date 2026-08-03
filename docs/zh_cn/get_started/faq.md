# 常见问题解答

- 如何启用 EpochBased 或者 IterBased 的训练？

  请查看此[教程](../common_usage/epoch_to_iter.md)。

- 验证集阶段和测试阶段的精度不一致？

  尝试训练时在 `model_wrapper_cfg` 中设置 `broadcast_buffer=True`。

  ```python
  model_wrapper_cfg = dict(
      type='MMDistributedDataParallel',
      broadcast_buffer=True,
  )
  ```
