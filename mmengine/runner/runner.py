# Copyright (c) OpenMMLab. All rights reserved.
import copy
import multiprocessing as mp
import os
import os.path as osp
import platform
import random
import shutil
import time
import warnings
from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config
from mmengine.data import worker_init_fn
from mmengine.dist import get_dist_info, init_dist, sync_random_seed
from mmengine.evaluator import BaseEvaluator
from mmengine.hooks import Hook
from mmengine.model import (MMDataParallel, MMDistributedDataParallel,
                            is_model_wrapper)
from mmengine.optim import _ParamScheduler, build_optimizer
from mmengine.registry import (DATA_SAMPLERS, DATASETS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS)
from mmengine.utils import is_list_of, symlink
from .base_loop import BaseLoop
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         get_state_dict, save_checkpoint, weights_to_cpu)
from .loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from .priority import Priority, get_priority


class Runner:
    """A training helper for PyTorch.

    TODO: Log related.

    Args:
        model (:obj:`torch.nn.Module` or dict): The model to be run.
        work_dir (str): The working directory to save checkpoints and logs.
        train_dataloader (Dataloader or dict, optional): An iterator to
            generate one batch of training dataset each iteration. If ``None``
            is given, it means skipping training steps. Defaults to None.
        val_dataloader (Dataloader or dict, optional): An iterator to generate
            one batch of validation dataset each iteration. If ``None`` is
            given, it means skipping validation steps. Defaults to None.
        test_dataloader (Dataloader or dict, optional): An iterator to generate
            one batch of test dataset each iteration. If ``None`` is
            given, it means skipping test steps. Defaults to None.
        train_cfg (dict, optional): A dict to build a training loop which is a
            subclass of :obj:`BaseLoop`. If specified, :attr:`train_dataloader`
            should also be specified . Defaults to None.
        val_cfg (dict, optional): A dict to build a validation loop which is a
            subclass of :obj:`BaseLoop`. If specified, :attr:`val_dataloader`
            should also be specified . Defaults to None.
        test_cfg (dict, optional): A dict to build a test loop which is a
            subclass of :obj:`BaseLoop`. If specified, :attr:`test_dataloader`
            should also be specified. Defaults to None.
        optimizer (Optimizer or dict, optional): Computing gradient of model
            parameters. If specified, :attr`train_dataloader` should also be
            specified. Defaults to None.
        param_scheduler (:obj:`_ParamScheduler` or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
        evaluator (:obj:`Evaluator` or dict or list, optional): Used for
            computing metrics. Defaults to None.
        default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
            execute default actions like updating model parameters and saving
            checkpoints. Default hooks have ``OptimizerHook``,
            ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook``,
            ``CheckpointHook``. Defaults to None.
        custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
            custom actions like visualizing images processed by pipeline.
            Defaults to None.
        load_checkpoint (dict, optional):
        launcher (str): Way to launcher multi processes. Supported launchers
            are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
            distributed training is disable.
        env_cfg (dict): A dict used for setting environment. Defaults to
            dict(dist_cfg=dict(backend='nccl')).
        log_cfg (dict, optional): A dict to build logger object. Defaults to
            None. TODO
        default_scope (str, optional): Used to reset registries location.
            Defaults to None.
        seed (int, optional): A number to guarantee reproducible results.
            If not specified, a random number will be set as seed. Defaults to
            None.
        cfg (:obj:`Config`, optional): Complete config. Defaults to None.
        deterministic (bool): Whether cudnn to select deterministic algorithms.
            See https://pytorch.org/docs/stable/notes/randomness.html.

    Examples:
        >>> from mmengine import Runner
        >>> cfg = dict(
                model=dict(type='ToyModel'),
                work_dir='path/of/work_dir',
                train_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=True),
                    batch_size=1,
                    num_workers=0),
                val_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=1,
                    num_workers=0),
                test_dataloader=dict(
                    dataset=dict(type='ToyDataset'),
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    batch_size=1,
                    num_workers=0),
                optimizer=dict(type='SGD', lr=0.01),
                param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
                evaluator=dict(type='ToyEvaluator'),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                val_cfg=dict(interval=1),
                test_cfg=dict(),
                custom_hooks=[],
                default_hooks=dict(
                    timer=dict(type='IterTimerHook'),
                    checkpoint=dict(type='CheckpointHook', interval=1),
                    logger=dict(type='TextLoggerHook'),
                    optimizer=dict(type='OptimizerHook', grad_clip=False),
                    param_scheduler=dict(type='ParamSchedulerHook')),
                env_cfg=dict(dist_cfg=dict(backend='nccl'), ),
                log_cfg=dict(log_level='INFO'),
            )
        >>> runner = Runner.build_from_cfg(cfg)
    """
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        cfg: Config,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        optimizer: Optional[Union[Optimizer, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        evaluator: Optional[Union[BaseEvaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        load_checkpoint: Optional[Dict] = None,
        launcher: Optional[str] = None,
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_cfg: Optional[dict] = None,
        default_scope: Optional[str] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # dump config
        self.cfg.dump(osp.join(self._work_dir, 'config.py'))

        # recursively copy the cfg because `self.cfg` will be modified
        # everywhere.
        self.cfg = copy.deepcopy(cfg)

        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        self.default_scope = default_scope

        # TODO, custom_imports

        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        # build a model
        if isinstance(model, dict):
            self.model = self.build_model(model)
        else:
            self.model = model

        # load model or resume training
        self._load_checkpoint = load_checkpoint
        if self._load_checkpoint is not None:
            if self._load_checkpoint['resume']:
                # self.resume(...)
                pass
            else:
                # self.load_checkpoint()
                pass

        if is_model_wrapper(
                self.model) and self.cfg.get('model_wrapper_cfg') is not None:
            raise TypeError(
                'model has been wrapped and "model_wrapper_cfg" should be None'
                f' but got {self.cfg.get("model_wrapper_cfg")}')

        if cfg.get('model_wrapper_cfg') is not None:
            self.model = self.wrap_model(
                cfg.get('model_wrapper_cfg'), self.model)

        # lazy initialization
        training_related = [
            train_dataloader, train_cfg, optimizer, param_scheduler
        ]
        if (not all(item is None for item in training_related)
                or not all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, optimizer, param_scheduler '
                'should be either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optimizer={optimizer}, '
                f'param_scheduler={param_scheduler}.')
        self.train_dataloader = train_dataloader
        self._train_loop = train_cfg
        self.optimizer = optimizer
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg]
        if (not all(item is None for item in val_related)
                or not all(item is not None for item in val_related)):
            raise ValueError(
                'val_dataloader and val_cfg should be either all None '
                f'or not None, but got val_dataloader={val_dataloader}, '
                f'val_cfg={val_cfg}')
        self.val_dataloader = val_dataloader
        self._val_loop = val_cfg

        test_related = [test_dataloader, test_cfg]
        if (not all(item is None for item in test_related)
                or not all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader and test_cfg should be either all None or not'
                f' None, but got test_dataloader={test_dataloader}, '
                f'test_cfg={test_cfg}')
        self.test_dataloader = test_dataloader
        self._test_loop = test_cfg

        if (self.val_dataloader is not None
                or self.test_dataloader is not None) and evaluator is None:
            raise ValueError(
                'evaluator should not None when val_dataloader or '
                'test_dataloader is not None.')
        self._evaluator = evaluator

        self._hooks: List[Hook] = []
        self.register_hooks(default_hooks, custom_hooks)

        self._launcher = launcher
        if self._launcher == 'none':
            self.distributed = False
        else:
            self.distributed = True

        self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.deterministic = deterministic
        self.seed = seed
        self.setup_env(env_cfg)

        self.log_cfg = log_cfg

        # TODO: self._exp_name
        # What type of information should added in _meta_info
        self._meta_info: dict = dict()

    @classmethod
    def build_from_cfg(cls, cfg: Config) -> 'Runner':
        """Build a runner from config dict.

        Args:
            cfg (:obj:`Config`): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        runner = cls(
            model=cfg.model,
            work_dir=cfg.work_dir,
            cfg=cfg,
            train_dataloader=cfg.train_dataloader,
            val_dataloader=cfg.val_dataloader,
            test_dataloader=cfg.test_dataloader,
            train_cfg=cfg.train_cfg,
            val_cfg=cfg.val_cfg,
            test_cfg=cfg.test_cfg,
            optimizer=cfg.optimizer,
            param_scheduler=cfg.param_scheduler,
            evaluator=cfg.evaluator,
            default_hooks=cfg.default_hooks,
            custom_hooks=cfg.custom_hooks,
            load_checkpoint=cfg.load_checkpoint,
            launcher=cfg.launcher,
            env_cfg=cfg.env_cfg,
            log_cfg=cfg.log_cfg,
            default_scope=cfg.default_scope,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
        )

        return runner

    @property
    def epoch(self):
        return self._epoch

    @property
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def meta_info(self):
        return self._meta_info

    def setup_env(self, env_cfg: Dict) -> None:
        """Setup environment.

        Args:
            env_cfg (dict): Config for setting environment.

        An example of ``env_cfg`` format:

        .. code-block:: python

            env_cfg = dict(
                cudnn_benchmark=True,
                mp_cfg=dict(
                    mp_start_method='fork',
                    opencv_num_threads=0
                ),
                dist_cfg=dict(backend='nccl'),
            )
        """
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self._set_multi_processing(**env_cfg.get('mp_cfg'))  # type: ignore

        # init distributed env first, since logger depends on the dist info.
        if self.distributed:
            init_dist(**env_cfg.get('dist_cfg'))  # type: ignore

        # set random seeds
        self._set_random_seed()

    def _set_multi_processing(self,
                              mp_start_method: str = 'fork',
                              opencv_num_threads: int = 0) -> None:
        """Set multi-processing related env.

        Args:
            mp_start_method (str): Set the method which should be used to start
                child processes. Defaults to 'fork'.
            opencv_num_threads (int): Number of threads for opencv.
                Defaults to 0.
        """
        # set multi-process start method as `fork` to speed up the training
        if platform.system() != 'Windows':
            current_method = mp.get_start_method(allow_none=True)
            if (current_method is not None
                    and current_method != mp_start_method):
                warnings.warn(
                    f'Multi-processing start method `{mp_start_method}` is '
                    f'different from the previous setting `{current_method}`.'
                    f'It will be force set to `{mp_start_method}`. You can '
                    'change this behavior by changing `mp_start_method` in '
                    'your config.')
            mp.set_start_method(mp_start_method, force=True)

        try:
            import cv2

            # disable opencv multithreading to avoid system being overloaded
            cv2.setNumThreads(opencv_num_threads)
        except ImportError:
            pass

        # setup OMP threads
        # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
        if 'OMP_NUM_THREADS' not in os.environ and self.distributed:
            omp_num_threads = 1
            warnings.warn(
                'Setting OMP_NUM_THREADS environment variable for each process'
                f' to be {omp_num_threads} in default, to avoid your system '
                'being overloaded, please further tune the variable for '
                'optimal performance in your application as needed.')
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

        # setup MKL threads
        if 'MKL_NUM_THREADS' not in os.environ and self.distributed:
            mkl_num_threads = 1
            warnings.warn(
                'Setting MKL_NUM_THREADS environment variable for each process'
                f' to be {mkl_num_threads} in default, to avoid your system '
                'being overloaded, please further tune the variable for '
                'optimal performance in your application as needed.')
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

    def _set_random_seed(self) -> None:
        """Set random seed to guarantee reproducible results.

        Warning:
            Results can not be guaranteed to resproducible if ``self.seed`` is
            None because :meth:`_set_random_seed` will generate a random seed.

        See https://pytorch.org/docs/stable/notes/randomness.html for details.
        """
        if self.seed is None:
            self.seed = sync_random_seed()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def build_model(self, model_cfg: Dict) -> nn.Module:
        """Build model.

        Args:
            model_cfg (dict): Config to build model.

        An example of ``model_cfg`` format:

        .. code-block:: python

            model_cfg = dict(type='ResNet')

        Returns:
            nn.Module: Model build from ``model_cfg``.
        """
        model = MODELS.build(model_cfg, default_scope=self.default_scope)

        if not hasattr(model, 'train_step'):
            # TODO, fix the url
            raise RuntimeError(
                'model contains at least `train_step` method. More details can'
                ' be found at TODO')

        return model

    def wrap_model(self, model_wrapper_cfg: Optional[Dict],
                   model: nn.Module) -> nn.Module:
        """Wrap model.

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``MMDistributedDataParallel`` or ``MMDataParallel``
                will be used. Defaults to None.

        An example of ``model_wrapper_cfg``:

        .. code-block:: python

            model_wrapper = dict(
                type='MMDistributedDataParallel',
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Returns:
            nn.Module: Wrapped model.
        """
        if model_wrapper_cfg is None:
            if self.distributed:
                find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                      False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = MMDistributedDataParallel(
                    self.model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                model = MMDataParallel(
                    model.cuda(self.cfg.gpu_ids[0]),
                    device_ids=self.cfg.gpu_ids)
        else:
            model = MODEL_WRAPPERS.build(
                model_wrapper_cfg,
                model=self.model,
                default_scope=self.default_scope)
        return model

    def build_optimizer(self, optimizer_cfg: Dict) -> Optimizer:
        """Build optimizer.

        Args:
            optimizer_cfg (dict): Config to build optimizer.

        An example of ``optimizer_cfg``:

        .. code-block:: python

            optimizer_cfg = dict(type='SGD', lr=0.01)

        Returns:
            Optimizer: Optimizer build from ``optimizer_cfg``.
        """
        # TODO, default scope
        optimizer = build_optimizer(self.model, optimizer_cfg)
        return optimizer

    def build_param_scheduler(
            self, scheduler_cfg: Union[Dict,
                                       List[Dict]]) -> List[_ParamScheduler]:
        """Build parameter schedulers.

        Args:
            scheduler_cfg (dict or list[dict]): Config to build parameter
                schedulers.

        An example of ``scheduler_cfg``:

        .. code-block:: python

            scheduler_cfg=dict(type='MultiStepLR', milestones=[1, 2]),

        Returns:
            list[:obj:`_ParamScheduler`]: Parameter schedulers build from
            ``scheduler_cfg``.
        """
        if not isinstance(self.optimizer, Optimizer):
            raise RuntimeError(
                'build_optimizer should be called early than '
                'build_param_scheduler because the latter depends on the '
                'former')

        if isinstance(scheduler_cfg, dict):
            scheduler_cfg = [scheduler_cfg]

        schedulers = []
        for cfg in scheduler_cfg:
            schedulers.append(
                PARAM_SCHEDULERS.build(
                    cfg,
                    optimizer=self.optimizer,
                    default_scope=self.default_scope))

        return schedulers

    def build_dataloader(self, dataloader_cfg: Dict) -> DataLoader:
        """Build dataloader.

        Args:
            dataloader_cfg (dict): A dict to build dataloader.

        An example of ``scheduler_cfg``:

        .. code-block:: python

            dataloader_cfg = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Returns:
            Dataloader: Dataloader build from ``dataloader_cfg``.
        """
        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        dataset = DATASETS.build(dataset_cfg)

        # build sampler
        rank, world_size = get_dist_info()
        sampler_cfg = dataloader_cfg.pop('sampler')
        sampler = DATA_SAMPLERS.build(
            sampler_cfg,
            default_args=dict(
                dataset=dataset, world_size=world_size, rank=rank))

        init_fn: Optional[partial]
        if self.seed is not None:
            init_fn = partial(
                worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=rank,
                seed=self.seed)
        else:
            init_fn = None

        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_sampler=None,
            # TODO
            # collate_fn=partial(collate, samples_per_gpu=dataloader_cfg.samples_per_gpu),  # noqa: E501
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader

    def build_train_loop(self, loop_cfg: Dict) -> BaseLoop:
        """Build training loop.

        Args:
            loop_cfg (dict): Config to build training loop.

        An example of ``loop_cfg``:

        .. code-block:: python

            loop_cfg = dict(by_epoch=True, max_epochs=3)

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop_cfg``.
        """
        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_scope=self.default_scope,
                dataloader=self.train_dataloader)
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, dataloader=self.train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, dataloader=self.train_dataloader)

        # `build_optimizer` should be called early than `build_param_scheduler`
        #  because the latter depends on the former
        if self.optimizer is not None and isinstance(self.optimizer, dict):
            self.optimizer = self.build_optimizer(self.optimizer)

        if (self.param_schedulers is not None
                and not is_list_of(self.param_schedulers, _ParamScheduler)):
            self.param_schedulers = self.build_param_scheduler(
                self.param_schedulers)  # type: ignore

        return loop

    def build_val_loop(self, loop_cfg: Dict) -> BaseLoop:
        """Build validating loop.

        Args:
            loop_cfg (dict): Config to build validating loop.

        An example of ``loop_cfg``:

        .. code-block:: python

            loop_cfg = dict(interval=1)

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop_cfg``.
        """
        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_scope=self.default_scope,
                dataloader=self.val_dataloader)
        else:
            loop = ValLoop(
                **loop_cfg,
                dataloader=self.val_dataloader,
                evaluator=self._evaluator)  # type: ignore

        return loop

    def build_test_loop(self, loop_cfg: Dict) -> BaseLoop:
        """Build test loop.

        Args:
            loop_cfg (dict): Config to build test loop.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_scope=self.default_scope,
                dataloader=self.test_dataloader)
        else:
            loop = TestLoop(
                **loop_cfg,
                dataloader=self.test_dataloader,
                evaluator=self._evaluator)  # type: ignore

        return loop

    def train(self) -> None:
        """Launch training."""
        assert self._train_loop is not None
        if not isinstance(self._train_loop, BaseLoop):
            self._train_loop = self.build_train_loop(self._train_loop)

        self._train_loop.run()  # type: ignore

    def val(self) -> None:
        """Launch validating."""
        assert self._val_loop is not None
        if not isinstance(self._val_loop, BaseLoop):
            self._val_loop = self.build_val_loop(self._val_loop)

        self._val_loop.run()  # type: ignore

    def test(self) -> None:
        """Launch test."""
        assert self._test_loop is not None
        if not isinstance(self._test_loop, BaseLoop):
            self._test_loop = self.build_test_loop(self._test_loop)

        self._test_loop.run()  # type: ignore

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments are passed to hook.
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self, **kwargs)

    def register_hook(
            self,
            hook: Union[Hook, Dict],
            priority: Optional[Union[str, int, Priority]] = None) -> None:
        """Register a hook into the hook list.

        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.

        Priority of hook will be decided with the following priority:

        - ``priority`` argument. If ``priority`` is given, it will be priority
          of hook.
        - If ``hook`` argument is a dict and ``priority`` in it, the priority
          will be the value of ``hook['priority']``.
        - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
          is an instance of ``hook``, the priority will be ``hook.priority``.

        Args:
            hook (:obj:`Hook` or dict): The hook to be registered.
            priority (int or str or :obj:`Priority`, optional): Hook priority.
                Lower value means higher priority.
        """
        if not isinstance(hook, (Hook, dict)):
            raise TypeError(
                f'hook should be an instance of Hook or dict, but got {hook}')

        _priority = None
        if isinstance(hook, dict):
            if 'priority' in hook:
                _priority = hook.pop('priority')

            hook_obj = HOOKS.build(hook)
        else:
            hook_obj = hook

        if priority is not None:
            hook_obj.priority = priority
        elif _priority is not None:
            hook_obj.priority = _priority

        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if get_priority(hook_obj.priority) >= get_priority(
                    self._hooks[i].priority):
                self._hooks.insert(i + 1, hook_obj)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook_obj)

    def register_default_hooks(self, optimizer: Union[Hook, Dict],
                               timer: Union[Hook, Dict], logger: Union[Hook,
                                                                       Dict],
                               param_scheduler: Union[Hook, Dict],
                               checkpoint: Union[Hook, Dict]) -> None:
        """Register default hooks into hook list.

        Args:
            hooks (dict[str, Hook or dict]): Set of hooks or configs to be
                registered.

        Default hooks and their priorities:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | OptimizerHook        | HIGH (30)               |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (40)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+
        """
        self.register_hook(optimizer)
        self.register_hook(timer)
        self.register_hook(logger)
        self.register_hook(param_scheduler)
        self.register_hook(checkpoint)

    def register_custom_hooks(self, hooks: List[Union[Hook, Dict]]) -> None:
        """Register custom hooks into hook list.

        Args:
            hooks (list[Hook | dict]): List of hooks or configs to be
                registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hooks(
            self,
            default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
            custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
        """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints. Default hooks have ``OptimizerHook``,
                ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook``,
                ``CheckpointHook``. Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        if default_hooks is not None:
            self.register_default_hooks(**default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def resume(self):
        """Resume training."""
        pass

    def load_checkpoint(self,
                        filename: str,
                        map_location: str = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str): A string specifying how to remap storage
                locations. Defaults to 'cpu'
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Default: strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_ckpt', checkpoint=checkpoint)

        return _load_checkpoint_to_model(
            self.model, checkpoint, strict, revise_keys=revise_keys)

    def save_checkpoint(self,
                        out_dir: str,
                        filename_tmpl: str = 'epoch_{}.pth',
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        meta: dict = None,
                        create_symlink: bool = True):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if self._meta_info is not None:
            meta.update(self._meta_info)
            # Note: meta.update(self._meta_info) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self._epoch + 1, iter=self._iter)

        # TODO, the filename passed by CheckpointHook
        filename = filename_tmpl.format(self._epoch + 1)
        filepath = osp.join(out_dir, filename)

        if hasattr(self.model, 'CLASSES') and self.model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=self.model.CLASSES)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model))
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optimizer, Optimizer):
                checkpoint['optimizer'] = self.optimizer.state_dict()
            elif isinstance(self.optimizer, dict):
                # TODO, current self.optimizer will not be a dict
                checkpoint['optimizer'] = {}
                for name, _optimizer in self.optimizer.items():
                    checkpoint['optimizer'][name] = _optimizer.state_dict()
            else:  # TODO
                raise TypeError(
                    'self.optimizer should be Optimizer of dict, but got '
                    f'{self.optimizer}')

        # save scheduler state dict
        if save_param_scheduler:
            checkpoint['scheduler'] = []
            for _scheduler in self.param_schedulers:  # type: ignore
                checkpoint['scheduler'].append(_scheduler.state_dict())

        self.call_hook('before_save_ckpt', checkpoint=checkpoint)

        save_checkpoint(checkpoint, filepath)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
