# 执行器（Runner）

在 MMEngine 中，我们抽象出了任务（Task）这个概念，例如模型的训练、测试、推理，都属于任务，而负责执行这些任务的模块就叫做执行器。

在介绍如何使用执行器之前，我们先举几个例子来帮助用户理解为什么需要执行器。

下面是一段使用 PyTorch 进行模型训练的伪代码：

```python
model = ResNet()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
train_dataset = ImageNetDataset(...)
train_dataloader = DataLoader(train_dataset, ...)

for i in range(max_epochs):
    for data_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(data_batch)
        loss = loss_func(outputs, data_batch)
        loss.backward()
        optimizer.step()
```

下面是一段使用 PyTorch 进行模型测试的伪代码：

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

inference_data =

for data_batch in test_dataloader:
    outputs = model(data_batch)
    acc = calculate_acc(outputs, data_batch)
```

下面是一段使用 PyTorch 进行模型推理的伪代码：

```python
model = ResNet()
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()

for img in image_list:
    prediction = model(img)
```

可以从上面的三段代码看出，这三个任务的执行流程都可以归纳为构建模型、读取数据、循环迭代等步骤。上述代码都是以图像分类为例，但不论是图像分类还是目标检测或是图像分割，都脱离不了这套范式。
因此，我们将模型的训练、验证、测试的流程整合起来，形成了执行器。在执行器中，我们只需要准备好模型、数据等任务必须的模块或是这些模块的配置文件，执行器会自动完成任务流程的准备。
通过使用执行器以及 MMEngine 中丰富的功能模块，用户不再需要手动搭建训练测试的流程，也不再需要去处理单 GPU 训练和分布式训练的区别，可以专心于算法和模型本身。

## 如何使用执行器

### 手动构建模块来构建执行器

### 通过配置文件构建执行器

### 使用执行器进行训练、测试和验证

## 自定义执行流程

## 自定义执行器
