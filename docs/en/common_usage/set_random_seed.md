# How to Set Random Seed

As described in [PyTorch REPRODUCIBILITY](https://pytorch.org/docs/stable/notes/randomness.html), there are  2 factors affecting the reproducibility of an experiment, namely random number and nondeterministic algorithms.

MMEngine provides the ability to set the random number and select a deterministic algorithm. Users can simply set the `randomness` parameter of the `Runner`, which eventually calls [set_random_seed](mmengine.runner.set_random_seed), and it has the following three fields:

- seed: The random seed. If this argument is not set, a random number will be used.
- differ_rand_seed: Whether to set different seeds for different processes by adding the `rank` (process index) to the seed.
- deterministic: Whether to set deterministic options for the CUDNN backend.

Let's take the [Get Started in 15 Minutes](../get_started/15_minutes.md) as an example to demonstrate how to set `randomness` in MMEngine.

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

However, there may still be some differences between any two experiments, even with the random number set and the deterministic algorithms chosen. Here we briefly explain why and where these fluctuations come from.

First, randomness not only exists in OpenMMLab algorithms like MMDetection. You may encounter it in most of the deep learning frameworks. The core reason is that the atomic operations in CUDA are unordered and random during parallel training.

To further acknowledge this, you can refer to this PyTorch official [document](https://pytorch.org/docs/stable/notes/randomness.html), which explains the uncertainties an algorithm encounters during different training sessions in PyTorch and how to control them. One is the generation of random numbers, which can be controlled by setting the seed in Numpy and PyTorch. The other is the implementation of certain operators, which can be locked using some switches in PyTorch, such as `torch.use_deterministic_algorithms`. Currently, the OpenMMLab series algorithms provide interfaces to use the same seed and the same implementations during different training sessions in the same algorithm. This is why users can see the "Set random seed to 0, deterministic: True" in the log during the training.

Then why do we still have fluctuations? This is because the CUDA implementation of some operators sometimes inevitably performs atomic operations such as adding, subtracting, multiplying, and dividing the same memory address multiple times in different CUDA kernels. In particular, during the `backward` process, the use of `atomicAdd` is very common. These atomic operations are unordered and random when computed. Therefore, when performing atomic operations at the same memory address multiple times, let's say adding multiple gradients at the same address, the order in which they are performed is uncertain, and even if each number is the same, the order in which the numbers are added will be different.

The randomness of the summing order leads to another problem, that is, since the summed values are generally floating point numbers that have the problem of precision loss, there will be a slight difference in the final result.

So, by setting random seeds and deterministic to `True`, we can make sure that the initialization weights and even the forward outputs of the model are identical for each experiment, and the loss values are also identical. However, there may be subtle differences after one backpropagation, and the final performance of the trained models will be slightly different.
