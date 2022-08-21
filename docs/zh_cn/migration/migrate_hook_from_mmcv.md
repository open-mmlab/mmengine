# 迁移 MMCV 钩子到 MMEngine

## 简介

由于架构设计的更新和用户需求的不断增加，MMCV 的钩子（Hook）点位已经满足不了需求，因此在 MMEngine 中对 Hook 点位进行了重新设计。在开始迁移前，阅读[钩子的设计](../design/hook.md)会很有帮助。

## 接口差异

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
</style>

<table class="tg" style="undefined;table-layout: fixed; width: 688px">
<colgroup>
<col style="width: 116px">
<col style="width: 131px">
<col style="width: 168px">
<col style="width: 273px">
</colgroup>
<thead>
  <tr>
    <th class="tg-9wq8" colspan="2"></th>
    <th class="tg-uzvj">MMCV Hook</th>
    <th class="tg-uzvj">MMEngine Hook</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">全局位点</td>
    <td class="tg-9wq8">执行前</td>
    <td class="tg-9wq8">before_run</td>
    <td class="tg-9wq8">before_run</td>
  </tr>
  <tr>
    <td class="tg-9wq8">执行后</td>
    <td class="tg-9wq8">after_run</td>
    <td class="tg-9wq8">after_run</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">Checkpoint 相关</td>
    <td class="tg-9wq8">加载 checkpoint 后</td>
    <td class="tg-9wq8">after_load_checkpoint</td>
    <td class="tg-9wq8">after_load_checkpoint</td>
  </tr>
  <tr>
    <td class="tg-9wq8">保存 checkpoint 前</td>
    <td class="tg-9wq8">before_save_checkpoint</td>
    <td class="tg-9wq8">before_save_checkpoint</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="6">训练相关</td>
    <td class="tg-9wq8">训练前触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">before_train</td>
  </tr>
  <tr>
    <td class="tg-9wq8">训练后触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">after_train</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 前</td>
    <td class="tg-9wq8">before_train_epoch</td>
    <td class="tg-9wq8">before_train_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 后</td>
    <td class="tg-9wq8">after_train_epoch</td>
    <td class="tg-9wq8">after_train_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代前</td>
    <td class="tg-9wq8">before_train_iter</td>
    <td class="tg-9wq8">before_train_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代后</td>
    <td class="tg-9wq8">after_train_iter</td>
    <td class="tg-9wq8">after_train_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="6">验证相关</td>
    <td class="tg-9wq8">验证前触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">before_val</td>
  </tr>
  <tr>
    <td class="tg-9wq8">验证后触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">after_val</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 前</td>
    <td class="tg-9wq8">before_val_epoch</td>
    <td class="tg-9wq8">before_val_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 后</td>
    <td class="tg-9wq8">after_val_epoch</td>
    <td class="tg-9wq8">after_val_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代前</td>
    <td class="tg-9wq8">before_val_iter</td>
    <td class="tg-9wq8">before_val_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代后</td>
    <td class="tg-9wq8">after_val_iter</td>
    <td class="tg-9wq8">after_val_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="6">测试相关</td>
    <td class="tg-9wq8">测试前触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">before_test</td>
  </tr>
  <tr>
    <td class="tg-9wq8">测试后触发</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">after_test</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 前</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">before_test_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每个 epoch 后</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">after_test_epoch</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代前</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">before_test_iter，新增 batch_idx 和 data_batch 参数</td>
  </tr>
  <tr>
    <td class="tg-9wq8">每次迭代后</td>
    <td class="tg-9wq8">无</td>
    <td class="tg-9wq8">after_test_iter，新增 batch_idx、data_batch 和 outputs 参数</td>
  </tr>
</tbody>
</table>

## 迁移示例

以 `CheckpointHook` 为例，MMEngine 的 [CheckpointHook](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py) 相比 MMCV 的 [CheckpointHook](https://github.com/open-mmlab/mmcv/blob/v1.6.0/mmcv/runner/hooks/checkpoint.py)（新增保存最优权重的功能（在 MMCV 中，保存最优权重的功能由 EvalHook 提供），因此，它需要实现 `after_val_epoch` 点位。

<table align="center">
  <thead>
      <tr align='center'>
          <td>MMCV CheckpintHook</td>
          <td>MMEngine CheckpointHook</td>
      </tr>
  </thead>
  <tbody><tr valign='top'>
  <th>

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""
```

</th>
  <th>

```python
class CheckpointHook(Hook):
    def before_run(self, runner):
        """初始化 out_dir 和 file_client 属性"""

    def after_train_epoch(self, runner):
        """同步 buffer 和保存权重，用于以 epoch 为单位训练的任务"""

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """同步 buffer 和保存权重，用于以 iteration 为单位训练的任务"""

    def after_val_epoch(self, runner, metrics):
        """根据 metrics 保存最优权重"""
```

</th></tr>
</tbody></table>
