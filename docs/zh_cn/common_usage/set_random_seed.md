# 设置随机种子

在 [PyTorch REPRODUCIBILITY](https://pytorch.org/docs/stable/notes/randomness.html) 中介绍了影响实验是否可复现的两个因素，一个是随机数，另一个是某些算子的实现算法具有不确定性。

MMEngine 提供了设置随机数以及是否选择确定性算法的功能，用户只需设置 `Runner` 的 `randomness` 参数（最终调用 [set_random_seed](mmengine.runner.set_random_seed)）即可，它有以下三个可设置的字段：

- seed: 随机种子，如果不设置 seed，则会使用随机数作为种子
- diff_rank_seed: 是否为不同的进程设置不同的种子，在 seed 的基础上加上进程索引数
- deterministic: 是否为 CUDNN 后端设置确定性选项

以[15 分钟上手 MMEngine](../get_started/15_minutes.md) 的 Runner 初始化参数中添加 `randomness` 为例。

```python
runner = Runner(
    model=MMResNet50(),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.001, momentum=0.9)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
    # 新增 randomness 设置
    randomness=dict(seed=0),
)
runner.train()
```

需要注意的是，即使设置了随机数以及选择了确定性算法，依然可能出现两次实验有波动，具体分析见[基于PyTorch的MMDetection中训练的随机性来自何处？](https://www.zhihu.com/question/453511684/answer/1839683634)
