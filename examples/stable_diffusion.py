# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

from mmengine import print_log
from mmengine.hooks import Hook
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner


class VisualizationHook(Hook):
    """Basic hook that invoke visualizers after train epoch.

    Args:
        prompt (`List[str]`):
                The prompts to guide the image generation.
    """
    priority = 'NORMAL'

    def __init__(self, prompt: List[str]):
        self.prompt = prompt

    def after_train_epoch(self, runner) -> None:
        images = runner.model.infer(self.prompt)
        for i, image in enumerate(images):
            runner.visualizer.add_image(
                f'image{i}_step', image, step=runner.epoch)


class MMStableDiffusion(BaseModel):
    """Stable Diffusion.

    Args:
        model (str): pretrained model name of stable diffusion.
            Defaults to 'runwayml/stable-diffusion-v1-5'.
        noise_offset_weight (bool, optional):
            The weight of noise offset introduced in
            https://www.crosslabs.org/blog/diffusion-with-offset-noise
            Defaults to 0.
    """

    def __init__(
        self,
        model: str = 'runwayml/stable-diffusion-v1-5',
        noise_offset_weight: float = 0,
    ):
        super().__init__()
        self.model = model

        self.enable_noise_offset = noise_offset_weight > 0
        self.noise_offset_weight = noise_offset_weight

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model, subfolder='tokenizer')
        self.scheduler = DDPMScheduler.from_pretrained(
            model, subfolder='scheduler')

        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder='text_encoder')
        self.vae = AutoencoderKL.from_pretrained(model, subfolder='vae')
        self.unet = UNet2DConditionModel.from_pretrained(
            model, subfolder='unet')
        self.prepare_model()

    def prepare_model(self):
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.vae.requires_grad_(False)
        print_log('Set VAE untrainable.', 'current')
        self.text_encoder.requires_grad_(False)
        print_log('Set Text Encoder untrainable.', 'current')

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def infer(self, prompt: List[str]) -> List[np.ndarray]:
        """Function invoked when calling the pipeline for generation.

        Args:
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
        """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            safety_checker=None,
        )
        pipeline.set_progress_bar_config(disable=True)
        images = []
        for p in prompt:
            image = pipeline(p, num_inference_steps=50).images[0]
            images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def train_step(self, data,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        data = self.data_preprocessor(data)

        latents = self.vae.encode(data['pixel_values']).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        if self.enable_noise_offset:
            noise = noise + self.noise_offset_weight * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=noise.device)

        num_batches = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps, (num_batches, ),
            device=self.device)
        timesteps = timesteps.long()

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(data['input_ids'])[0]

        if self.scheduler.config.prediction_type == 'epsilon':
            gt = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError('Unknown prediction type '
                             f'{self.scheduler.config.prediction_type}')

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        loss_dict = dict()
        # calculate loss in FP32
        loss_mse = F.mse_loss(model_pred.float(), gt.float())
        loss_dict['loss_mse'] = loss_mse

        parsed_loss, log_vars = self.parse_losses(loss_dict)
        optim_wrapper.update_params(parsed_loss)

        return log_vars

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def create_dataset(tokenizer,
                   image_column='image',
                   caption_column='text',
                   resolution=512,
                   center_crop=True,
                   random_flip=True):
    dataset = load_dataset('lambdalabs/pokemon-blip-captions')
    column_names = dataset['train'].column_names
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{image_column}' needs to be one of:"
            f" {', '.join(column_names)}")
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{caption_column}' needs to be one of:"
            f" {', '.join(column_names)}")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(
                    random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f'Caption column `{caption_column}` should contain'
                    ' either strings or lists of strings.')
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose([
        transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution)
        if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip()
        if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert('RGB') for image in examples[image_column]]
        examples['pixel_values'] = [
            train_transforms(image) for image in images
        ]
        examples['input_ids'] = tokenize_captions(examples)
        return examples

    train_dataset = dataset['train'].with_transform(preprocess_train)
    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack(
        [example['pixel_values'] for example in examples])
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example['input_ids'] for example in examples])
    return {'pixel_values': pixel_values, 'input_ids': input_ids}


def main():
    args = parse_args()
    model = MMStableDiffusion()

    dataset = create_dataset(model.tokenizer)
    train_loader = dict(
        batch_size=4,
        dataset=dataset,
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=collate_fn)
    default_hooks = dict(
        checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3), )
    custom_hooks = [VisualizationHook(prompt=['yoda pokemon'] * 4)]
    runner = Runner(
        model=model,
        train_dataloader=train_loader,
        optim_wrapper=dict(
            optimizer=dict(type=torch.optim.AdamW, lr=1e-5, weight_decay=1e-2),
            clip_grad=dict(max_norm=1.0)),
        train_cfg=dict(by_epoch=True, max_epochs=50),
        launcher=args.launcher,
        work_dir='work_dirs',
        default_hooks=default_hooks,
        custom_hooks=custom_hooks,
    )
    runner.train()


if __name__ == '__main__':
    main()
