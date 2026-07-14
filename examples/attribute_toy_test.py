import torch
from torch import nn
from torch.utils.data import DataLoader

from mmengine.model import BaseModel
from mmengine.runner import Runner


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_sample = torch.stack(data_samples)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_sample - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


x = [(torch.ones(2, 2), [torch.ones(2, 1)])]
# train_dataset = [x, x, x]
train_dataset = x * 50
train_dataloader = DataLoader(train_dataset, batch_size=1)

runner = Runner(
    model=ToyModel(),
    custom_hooks=[
        dict(
            type='RecorderHook',
            recorders=[
                dict(type='AttributeRecorder', target='self.linear1.weight')
            ],
            save_dir='./work_dir',
            print_modification=True)
    ],
    work_dir='tmp_dir',
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=10),
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.01)))
runner.train()
