# Copyright (c) OpenMMLab. All rights reserved.
import open_clip
from torch.optim import AdamW

from mmengine.model import BaseModel
from mmengine.runner import Runner
from .data import get_data
from .params import parse_args
from .scheduler import cosine_lr


class MMCLIP(BaseModel):

    def __init__(self, model):
        super().__init__()
        self.clip = model
        self.criterion = open_clip.ClipLoss(cache_labels=True)

    def forward(self, imgs, texts, mode):
        image_features, text_features, logit_scale = self.clip(imgs, texts)
        if mode == 'loss':
            return {
                'loss': self.criterion(image_features, text_features,
                                       logit_scale)
            }
        elif mode == 'predict':
            return image_features, text_features


def build_optimizer(model, args):
    exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or 'bias' in n or 'logit_scale' in n  # noqa: E731,E501
    include = lambda n, p: not exclude(n, p)  # noqa: E731

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [
        p for n, p in named_parameters if include(n, p) and p.requires_grad
    ]

    optimizer = AdamW(
        [
            {
                'params': gain_or_bias_params,
                'weight_decay': 0.
            },
            {
                'params': rest_params,
                'weight_decay': args.wd
            },
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    return optimizer


def main():
    args = parse_args()
    # build model
    model, preprocess_train, preprocess_val = \
        open_clip.create_model_and_transforms(model_name=args.model,
                                              precision=args.precision,
                                              device='cuda:0')

    # build dataloader
    data = get_data(args, (preprocess_train, preprocess_val), epoch=0)
    train_dataloader = data['train'].dataloader

    # build optimizer(hypperparameter is valid for resnet)
    optimizer = build_optimizer(model, args)

    # build lr_scheduler
    total_steps = data['train'].dataloader.num_batches * args.epochs
    lr_scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # build runner
    runner = Runner(
        model=MMCLIP(model),
        optimizer_wrapper=dict(optimizer=optimizer),
        param_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        launcher=args.launcher,
    )
    runner.train()


if __name__ == '__main__':
    main()
