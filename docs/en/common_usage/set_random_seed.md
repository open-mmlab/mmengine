# How to Set Random Seed

There are two factors are introduced that affect whether an experiment is reproducible in [PyTorch REPRODUCIBILITY](https://pytorch.org/docs/stable/notes/randomness.html). One is the random number, and the other is the uncertainty of the implementation of certain operators.

MMEngine provides the ability to set the random number and select a deterministic algorithm. Users can simply set the `randomness` parameter of the `Runner`, which eventually calls [set_random_seed](mmengine.runner.set_random_seed), and it has the following three fields:

- seed: the random seed, a random number will be used as the seed if `seed` is not set.
- differ_rand_seed: whether to set different seeds for different processes by adding the process index number to the seed.
- deterministic: whether to set deterministic options for the CUDNN backend.


Let's take the [Get Started in 15 Minutes](../get_started/15_minutes.md) as an example to demonstrate how to add `randomness` in MMEngine.

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
    # adding randomness setting
    randomness=dict(seed=0),
)
runner.train()
```

Note: Even with the random number set and the deterministic algorithm chosen, there may still be fluctuations between any two experiments, as described in [Where does the randomness of training come from in PyTorch-based MMDetection?] (https://www.zhihu.com/question/453511684/answer/1839683634)
