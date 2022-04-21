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
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.data import pseudo_collate, worker_init_fn
from mmengine.dist import (broadcast, get_dist_info, init_dist, master_only,
                           sync_random_seed)
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger
from mmengine.model import is_model_wrapper
from mmengine.optim import _ParamScheduler, build_optimizer
from mmengine.registry import (DATA_SAMPLERS, DATASETS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               DefaultScope)
from mmengine.utils import (TORCH_VERSION, digit_version,
                            find_latest_checkpoint, is_list_of, symlink)
from mmengine.visualization import ComposedWriter
from .base_loop import BaseLoop
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         get_state_dict, save_checkpoint, weights_to_cpu)
from .loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from .priority import Priority, get_priority

ConfigType = Union[Dict, Config, ConfigDict]


class Runner:
    """A training helper for PyTorch.

    Runner object can be built from config by ``runner = Runner.from_cfg(cfg)``
    where the ``cfg`` usually contains training, validation, and test-related
    configurations to build corresponding components. We usually use the
    same config to launch training, testing, and validation tasks. However,
    only some of these components are necessary at the same time, e.g.,
    testing a model does not need training or validation-related components.

    To avoid repeatedly modifying config, the construction of ``Runner`` adopts
    lazy initialization to only initialize components when they are going to be
    used. Therefore, the model is always initialized at the beginning, and
    training, validation, and, testing related components are only initialized
    when calling ``runner.train()``, ``runner.val()``, and ``runner.test()``,
    respectively.

    Args:
        model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
            a dict used for build a model.
        work_dir (str): The working directory to save checkpoints and logs.
        train_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping training steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        val_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping validation steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        test_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping test steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        train_cfg (dict, optional): A dict to build a training loop. If it does
            not provide "type" key, it should contain "by_epoch" to decide
            which type of training loop :class:`EpochBasedTrainLoop` or
            :class:`IterBasedTrainLoop` should be used. If ``train_cfg``
            specified, :attr:`train_dataloader` should also be specified.
            Defaults to None. See :meth:`build_train_loop` for more details.
        val_cfg (dict, optional): A dict to build a validation loop. If it does
            not provide "type" key, :class:`ValLoop` will be used by default.
            If ``val_cfg`` specified, :attr:`val_dataloader` should also be
            specified. Defaults to None.
            See :meth:`build_val_loop` for more etails.
        test_cfg (dict, optional): A dict to build a test loop. If it does
            not provide "type" key, :class:`TestLoop` will be used by default.
            If ``test_cfg`` specified, :attr:`test_dataloader` should also be
            specified. Defaults to None.
            See :meth:`build_test_loop` for more etails.
        optimizer (Optimizer or dict, optional): Computing gradient of model
            parameters. If specified, :attr:`train_dataloader` should also be
            specified. Defaults to None.
        param_scheduler (_ParamScheduler or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
        val_evaluator (Evaluator or dict or list, optional): A evaluator object
            used for computing metrics for validation. It can be a dict or a
            list of dict to build a evaluator. If specified,
            :attr:`val_dataloader` should also be specified. Defaults to None.
        test_evaluator (Evaluator or dict or list, optional): A evaluator
            object used for computing metrics for test steps. It can be a dict
            or a list of dict to build a evaluator. If specified,
            :attr:`test_dataloader` should also be specified. Defaults to None.
        default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
            execute default actions like updating model parameters and saving
            checkpoints. Default hooks are ``OptimizerHook``,
            ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
            ``CheckpointHook``. Defaults to None.
            See :meth:`register_default_hooks` for more details.
        custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
            custom actions like visualizing images processed by pipeline.
            Defaults to None.
        load_from (str, optional): The checkpoint file to load from.
            Defaults to None.
        resume (bool): Whether to resume training. Defaults to False. If
            ``resume`` is True and ``load_from`` is None, automatically to
            find latest checkpoint from ``work_dir``. If not found, resuming
            does nothing.
        launcher (str): Way to launcher multi-process. Supported launchers
            are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
            non-distributed environment will be launched.
        env_cfg (dict): A dict used for setting environment. Defaults to
            dict(dist_cfg=dict(backend='nccl')).
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        writer (ComposedWriter or dict, optional): A ComposedWriter object or a
            dict build ComposedWriter object. Defaults to None. If not
            specified, default config will be used.
        default_scope (str, optional): Used to reset registries location.
            Defaults to None.
        randomness (dict): Some settings to make the experiment as reproducible
            as possible like seed and deterministic.
            Defaults to ``dict(seed=None)``. If seed is None, a random number
            will be generated and it will be broadcasted to all other processes
            if in distributed environment. If ``cudnn_benchmarch`` is
            ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
            ``randomness``, the value of ``torch.backends.cudnn.benchmark``
            will be ``False`` finally.
        experiment_name (str, optional): Name of current experiment. If not
            specified, timestamp will be used as ``experiment_name``.
            Defaults to None.
        cfg (dict or Configdict or :obj:`Config`, optional): Full config.
            Defaults to None.

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
                val_evaluator=dict(type='ToyEvaluator'),
                test_evaluator=dict(type='ToyEvaluator'),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                val_cfg=dict(interval=1),
                test_cfg=dict(),
                custom_hooks=[],
                default_hooks=dict(
                    timer=dict(type='IterTimerHook'),
                    checkpoint=dict(type='CheckpointHook', interval=1),
                    logger=dict(type='LoggerHook'),
                    optimizer=dict(type='OptimizerHook', grad_clip=False),
                    param_scheduler=dict(type='ParamSchedulerHook')),
                launcher='none',
                env_cfg=dict(dist_cfg=dict(backend='nccl')),
                writer=dict(
                    name='composed_writer',
                    writers=[dict(type='LocalWriter', save_dir='temp_dir')])
            )
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
        >>> runner.test()
    """
    cfg: ConfigType
    train_loop: Optional[Union[BaseLoop, Dict]]
    val_loop: Optional[Union[BaseLoop, Dict]]
    test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        optimizer: Optional[Union[Optimizer, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_level: str = 'INFO',
        writer: Optional[Union[ComposedWriter, Dict]] = None,
        default_scope: Optional[str] = None,
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            self.cfg = copy.deepcopy(cfg)
        else:
            self.cfg = dict()

        self._epoch = 0
        self._iter = 0

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optimizer]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optimizer should be either '
                'all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optimizer={optimizer}.')
        self.train_dataloader = train_dataloader
        self.train_loop = train_cfg
        self.optimizer = optimizer

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optimizer is None:
            raise ValueError(
                'param_scheduler should be None when optimizer is None, '
                f'but got {param_scheduler}')
        if not isinstance(param_scheduler, Sequence):
            self.param_schedulers = [param_scheduler]
        else:
            self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self.val_dataloader = val_dataloader
        self.val_loop = val_cfg
        self.val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self.test_dataloader = test_dataloader
        self.test_loop = test_cfg
        self.test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.get('filename') is not None:
            filename_no_ext = osp.splitext(osp.basename(
                self.cfg['filename']))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp

        self.logger = self.build_logger(log_level=log_level)
        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # writer used for writing log or visualizing all kinds of data
        self.writer = self.build_writer(writer)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        self.default_scope = DefaultScope.get_instance(
            self._experiment_name, scope_name=default_scope)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        self.model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)

        self.meta: dict = dict()

        # dump `cfg` to `work_dir`
        self.dump_config()

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            optimizer=cfg.get('optimizer'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_level=cfg.get('log_level', 'INFO'),
            writer=cfg.get('writer'),
            default_scope=cfg.get('default_scope'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner

    @property
    def experiment_name(self):
        """str: Name of experiment."""
        return self._experiment_name

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def work_dir(self):
        """str: The working directory to save checkpoints and logs."""
        return self._work_dir

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        """Update epoch and synchronize epoch in :attr:`message_hub`."""
        self._epoch = epoch
        # To allow components that cannot access runner to get current epoch.
        self.message_hub.update_info('epoch', epoch)

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @iter.setter
    def iter(self, iter: int):
        """Update iter and synchronize iter in :attr:`message_hub`."""
        self._iter = iter
        # To allow components that cannot access runner to get current
        # iteration.
        self.message_hub.update_info('iter', iter)

    @property
    def launcher(self):
        """str: Way to launcher multi processes."""
        return self._launcher

    @property
    def distributed(self):
        """bool: Whether current environment is distributed."""
        return self._distributed

    @property
    def rank(self):
        """int: Rank of current process."""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job."""
        return self._world_size

    @property
    def deterministic(self):
        """int: Whether cudnn to select deterministic algorithms."""
        return self._deterministic

    @property
    def seed(self):
        """int: A number to set random modules."""
        return self._seed

    @property
    def timestamp(self):
        """str: Timestamp when creating experiment."""
        return self._timestamp

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    def setup_env(self, env_cfg: Dict) -> None:
        """Setup environment.

        An example of ``env_cfg``::

            env_cfg = dict(
                cudnn_benchmark=True,
                mp_cfg=dict(
                    mp_start_method='fork',
                    opencv_num_threads=0
                ),
                dist_cfg=dict(backend='nccl'),
            )

        Args:
            env_cfg (dict): Config for setting environment.
        """
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        if env_cfg.get('mp_cfg') is not None:
            self._set_multi_processing(**env_cfg.get('mp_cfg'))  # type: ignore

        # init distributed env first, since logger depends on the dist info.
        if self.distributed and env_cfg.get('dist_cfg') is not None:
            init_dist(self.launcher, **env_cfg.get('dist_cfg'))  # type: ignore

        self._rank, self._world_size = get_dist_info()

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        # TODO: handled by broadcast
        if self._world_size > 1 and torch.cuda.is_available():
            timestamp = timestamp.cuda()
        # broadcast timestamp from 0 process to other processes
        broadcast(timestamp)
        self._timestamp = time.strftime('%Y%m%d_%H%M%S',
                                        time.localtime(timestamp.item()))

    def _set_multi_processing(self,
                              mp_start_method: str = 'fork',
                              opencv_num_threads: int = 0) -> None:
        """Set multi-processing related environment.

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

    def set_randomness(self, seed, deterministic: bool = False) -> None:
        """Set random seed to guarantee reproducible results.

        Args:
            seed (int): A number to set random modules.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Defaults to False.
                See https://pytorch.org/docs/stable/notes/randomness.html for
                more details.
        """
        self._deterministic = deterministic
        self._seed = seed
        if self._seed is None:
            self._seed = sync_random_seed()

        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        if deterministic:
            if torch.backends.cudnn.benchmark:
                warnings.warn(
                    'torch.backends.cudnn.benchmark is going to be set as '
                    '`False` to cause cuDNN to deterministically select an '
                    'algorithm')

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
                torch.use_deterministic_algorithms(True)

    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_file: str = None,
                     **kwargs) -> MMLogger:
        """Build a global asscessable MMLogger.

        Args:
            log_level (int or str): The log level of MMLogger handlers.
                Defaults to 'INFO'.
            log_file (str, optional): Path of filename to save log.
                Defaults to None.
            **kwargs: Remaining parameters passed to ``MMLogger``.

        Returns:
            MMLogger: A MMLogger object build from ``logger``.
        """
        if log_file is None:
            log_file = osp.join(self.work_dir, f'{self._experiment_name}.log')

        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', self._experiment_name)

        return MMLogger.get_instance(**log_cfg)  # type: ignore

    def build_message_hub(self,
                          message_hub: Optional[Dict] = None) -> MessageHub:
        """Build a global asscessable MessageHub.

        Args:
            message_hub (dict, optional): A dict to build MessageHub object.
                If not specified, default config will be used to build
                MessageHub object. Defaults to None.

        Returns:
            MessageHub: A MessageHub object build from ``message_hub``.
        """
        if message_hub is None:
            message_hub = dict(name=self._experiment_name)
        elif isinstance(message_hub, dict):
            # ensure message_hub containing name key
            message_hub.setdefault('name', self._experiment_name)
        else:
            raise TypeError(
                f'message_hub should be dict or None, but got {message_hub}')

        return MessageHub.get_instance(**message_hub)

    def build_writer(
        self,
        writer: Optional[Union[ComposedWriter,
                               Dict]] = None) -> ComposedWriter:
        """Build a global asscessable ComposedWriter.

        Args:
            writer (ComposedWriter or dict, optional): A ComposedWriter object
                or a dict to build ComposedWriter object. If ``writer`` is a
                ComposedWriter object, just returns itself. If not specified,
                default config will be used to build ComposedWriter object.
                Defaults to None.

        Returns:
            ComposedWriter: A ComposedWriter object build from ``writer``.
        """
        if isinstance(writer, ComposedWriter):
            return writer
        elif writer is None:
            writer = dict(
                name=self._experiment_name,
                writers=[dict(type='LocalWriter', save_dir=self._work_dir)])
        elif isinstance(writer, dict):
            # ensure writer containing name key
            writer.setdefault('name', self._experiment_name)
        else:
            raise TypeError(
                'writer should be ComposedWriter object, a dict or None, '
                f'but got {writer}')

        return ComposedWriter.get_instance(**writer)

    def build_model(self, model: Union[nn.Module, Dict]) -> nn.Module:
        """Build model.

        An example of ``model``::

            model = dict(type='ResNet')

        Args:
            model (nn.Module or dict): A nn.Module object or a dict to build
                nn.Module object. If ``model`` is a nn.Module object, just
                returns itself.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            return MODELS.build(model)
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def wrap_model(self, model_wrapper_cfg: Optional[Dict],
                   model: nn.Module) -> nn.Module:
        """Wrap model.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``DistributedDataParallel`` will be used in
                distributed environment. Defaults to None.
            model (nn.Module): Model to be wrapped.

        Returns:
            nn.Module: Wrapped model.
        """
        if is_model_wrapper(model):
            if model_wrapper_cfg is not None:
                raise TypeError(
                    'model has been wrapped and "model_wrapper_cfg" should be '
                    f'None, but got {model_wrapper_cfg}')

            return model

        if model_wrapper_cfg is None:
            if self.distributed:
                find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                      False)
                # Sets the `find_unused_parameters` parameter in
                # torch.nn.parallel.DistributedDataParallel
                model = DistributedDataParallel(
                    self.model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False,
                    find_unused_parameters=find_unused_parameters)
            else:
                # Set `export CUDA_VISIBLE_DEVICES=-1` can enable CPU training.
                if torch.cuda.is_available():
                    model = model.cuda()
        else:
            model = MODEL_WRAPPERS.build(
                model_wrapper_cfg, default_args=dict(model=self.model))

        return model

    def build_optimizer(self, optimizer: Union[Optimizer, Dict]) -> Optimizer:
        """Build optimizer.

        An example of ``optimizer``::

            optimizer = dict(type='SGD', lr=0.01)

        Args:
            optimizer (Optimizer or dict): An Optimizer object or a dict to
                build Optimizer object. If ``optimizer`` is an Optimizer
                object, just returns itself.

        Returns:
            Optimizer: Optimizer build from ``optimizer_cfg``.
        """
        if isinstance(optimizer, Optimizer):
            return optimizer
        elif isinstance(optimizer, dict):
            optimizer = build_optimizer(self.model, optimizer)
            return optimizer
        else:
            raise TypeError('optimizer should be an Optimizer object or dict, '
                            f'but got {optimizer}')

    def build_param_scheduler(
        self, scheduler: Union[_ParamScheduler, Dict,
                               List]) -> List[_ParamScheduler]:
        """Build parameter schedulers.

        Examples of ``scheduler``::

            scheduler = dict(type='MultiStepLR', milestones=[1, 2])

            # scheduler can also be a list of dict
            scheduler = [
                dict(type='MultiStepLR', milestones=[1, 2]),
                dict(type='StepLR', step_size=1)
            ]

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.

        Returns:
            list[:obj:`_ParamScheduler`]: List of parameter schedulers build
            from ``scheduler``.
        """
        if not isinstance(self.optimizer, Optimizer):
            raise RuntimeError(
                '`build_optimizer` should be called before'
                '`build_param_scheduler` because the latter depends on the '
                'former')

        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        param_schedulers = []
        for _scheduler in schedulers:
            if isinstance(_scheduler, _ParamScheduler):
                param_schedulers.append(_scheduler)
            elif isinstance(_scheduler, dict):
                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(optimizer=self.optimizer)))
            else:
                raise TypeError(
                    '_scheduler should be a _ParamScheduler object or dict, '
                    f'but got {_scheduler}')

        return param_schedulers

    def build_evaluator(
            self, evaluator: Union[Dict, List[Dict], Evaluator]) -> Evaluator:
        """Build evaluator.

        Examples of ``evaluator``::

            evaluator = dict(type='ToyMetric')

            # evaluator can also be a list of dict
            evaluator = [
                dict(type='ToyMetric1'),
                dict(type='ToyEvaluator2')
            ]

        Args:
            evaluator (Evaluator or dict or list):
                An Evaluator object or a config dict or list of config dict
                used to build an Evaluator.

        Returns:
            Evaluator: Evaluator build from ``evaluator``.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError(
                'evaluator should be one of dict, list of dict, and Evaluator'
                f', but got {evaluator}')

    def build_dataloader(self, dataloader: Union[DataLoader,
                                                 Dict]) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.

        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler = DATA_SAMPLERS.build(
                sampler_cfg, default_args=dict(dataset=dataset))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build dataloader
        init_fn: Optional[partial]
        if self.seed is not None:
            init_fn = partial(
                worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=self.rank,
                seed=self.seed)
        else:
            init_fn = None

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, to make this more flexible, collate_fn in MMengine does
        # nothing. The action to merge a list of samples will be handled
        # in model.
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_sampler=None,
            collate_fn=pseudo_collate,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self, dataloader=self.train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self.train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self.train_dataloader)

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optimizer = self.build_optimizer(self.optimizer)

        self.param_schedulers = self.build_param_scheduler(  # type: ignore
            self.param_schedulers)  # type: ignore

        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict(interval=1)

            # custom validation loop
            loop = dict(type='CustomValLoop', interval=1)

        Args:
            loop (BaseLoop or dict): A validation loop or a dict to build
                validation loop. If ``loop`` is a validation loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Validation loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self.val_dataloader,
                    evaluator=self.val_evaluator))
        else:
            loop = ValLoop(
                runner=self,
                dataloader=self.val_dataloader,
                evaluator=self.val_evaluator,  # type: ignore
                **loop_cfg,
            )  # type: ignore

        return loop  # type: ignore

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test loop.

        Examples of ``loop``:

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

        Args:
            loop_cfg (dict): Config to build test loop.

        Returns:
            :obj:`BaseLoop`: Test loop object build from ``loop_cfg``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)  # type: ignore

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self.test_dataloader,
                    evaluator=self.test_evaluator))
        else:
            loop = TestLoop(
                runner=self,
                dataloader=self.test_dataloader,
                evaluator=self.test_evaluator)  # type: ignore

        return loop  # type: ignore

    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            resume_from = find_latest_checkpoint(self.work_dir)

        if resume_from is not None:
            self.resume(resume_from)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True

    def train(self) -> None:
        """Launch training."""
        if self.train_loop is None:
            raise RuntimeError(
                '`self.train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self.train_loop = self.build_train_loop(
            self.train_loop)  # type: ignore

        if self.val_loop is not None:
            self.val_loop = self.build_val_loop(self.val_loop)  # type: ignore

        self.load_or_resume()

        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')
        self.train_loop.run()  # type: ignore
        self.call_hook('after_run')

    def val(self) -> None:
        """Launch validation."""
        if self.val_loop is None:
            raise RuntimeError(
                '`self.val_loop` should not be None when calling val method.'
                'Please provide `val_dataloader`, `val_cfg` and '
                '`val_evaluator` arguments when initializing runner.')

        self.val_loop = self.build_val_loop(self.val_loop)  # type: ignore

        self.load_or_resume()

        self.call_hook('before_run')
        self.val_loop.run()  # type: ignore
        self.call_hook('after_run')

    def test(self) -> None:
        """Launch test."""
        if self.test_loop is None:
            raise RuntimeError(
                '`self.test_loop` should not be None when calling test method.'
                'Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self.test_loop = self.build_test_loop(self.test_loop)  # type: ignore

        self.load_or_resume()

        self.call_hook('before_run')
        self.test_loop.run()  # type: ignore
        self.call_hook('after_run')

    def call_hook(self, fn_name: str, **kwargs) -> None:
        """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
        for hook in self._hooks:
            # support adding additional custom hook methods
            if hasattr(hook, fn_name):
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

    def register_default_hooks(
            self,
            hooks: Optional[Dict[str, Union[Hook, Dict]]] = None) -> None:
        """Register default hooks into hook list.

        ``hooks`` will be registered into runner to execute some default
        actions like updating model parameters or saving checkpoints.

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

        If ``hooks`` is None, above hooks will be registered by
        default::

            default_hooks = dict(
                optimizer=dict(type='OptimizerHook', grad_clip=None),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`OptimizerHook`,
        :obj:`LoggerHook`, :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            optimizer=dict(type='OptimizerHook', grad_clip=None),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1),
        )
        if hooks is not None:
            for name, hook in hooks.items():
                if name in default_hooks and hook is None:
                    # remove hook from _default_hooks
                    default_hooks.pop(name)
                else:
                    assert hook is not None
                    default_hooks[name] = hook

        for hook in default_hooks.values():
            self.register_hook(hook)

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
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
        self.register_default_hooks(default_hooks)

        if custom_hooks is not None:
            self.register_custom_hooks(custom_hooks)

    def resume(self,
               filename: str,
               resume_optimizer: bool = True,
               resume_param_scheduler: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    filename,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(filename)
        else:
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']

        if self.meta is None:
            self.meta = {}

        self.meta.setdefault('hook_msgs', {})
        # load `last_ckpt`, `best_score`, `best_ckpt`, etc. for hook messages
        self.meta['hook_msgs'].update(checkpoint['meta'].get('hook_msgs', {}))

        # check whether the number of GPU used for current experiment
        # is consistent with resuming from checkpoint
        if 'config' in checkpoint['meta']:
            config = mmengine.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
                    and len(previous_gpu_ids) != self._world_size):
                # TODO, should we modify the iteration?
                self.logger.info(
                    'Number of GPU used for current experiment is not '
                    'consistent with resuming from checkpoint')

        # resume meta information meta
        self.meta = checkpoint['meta']

        # resume optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer = self.build_optimizer(self.optimizer)
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # resume param scheduler
        if 'param_schedulers' in checkpoint and resume_param_scheduler:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)

            for cur_scheduler, ckpt_scheduler in zip(
                    self.param_schedulers, checkpoint['param_schedulers']):
                cur_scheduler.load_state_dict(ckpt_scheduler)  # type: ignore

        self._has_loaded = True

        self.logger.info(f'resumed epoch: {self._epoch}, iter: {self._iter}')

    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
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

        checkpoint = _load_checkpoint_to_model(
            self.model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint

    @master_only
    def save_checkpoint(self,
                        out_dir: str,
                        filename: str,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        meta: dict = None,
                        create_symlink: bool = True,
                        by_epoch: bool = True):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
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

        if self.meta is not None:
            meta.update(self.meta)

        if by_epoch:
            # self._epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self_epoch + 1`
            meta.update(epoch=self._epoch + 1, iter=self._iter)
        else:
            meta.update(epoch=self._epoch, iter=self._iter + 1)

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
            else:  # TODO
                raise TypeError(
                    'self.optimizer should be an optimizer, but got '
                    f'{self.optimizer}')

        # save param scheduler state dict
        if save_param_scheduler:
            checkpoint['param_schedulers'] = []
            for _scheduler in self.param_schedulers:
                state_dict = _scheduler.state_dict()  # type: ignore
                checkpoint['param_schedulers'].append(state_dict)

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

    @master_only
    def dump_config(self) -> None:
        """Dump config to `work_dir`."""
        if isinstance(self.cfg,
                      Config) and self.cfg.get('filename') is not None:
            self.cfg.dump(
                osp.join(self.work_dir, osp.basename(self.cfg.filename)))
        elif self.cfg:
            # TODO
            pass
