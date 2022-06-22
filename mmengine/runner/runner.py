# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import platform
import random
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.data import pseudo_collate, worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           master_only, sync_random_seed)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.logging import LogProcessor, MessageHub, MMLogger
from mmengine.model import (BaseModel, MMDistributedDataParallel,
                            is_model_wrapper)
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, HOOKS,
                               LOOPS, MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope,
                               count_registered_modules)
from mmengine.registry.root import LOG_PROCESSORS
from mmengine.utils import (TORCH_VERSION, digit_version, get_git_hash,
                            is_list_of, revert_sync_batchnorm,
                            set_multi_processing)
from mmengine.visualization import Visualizer
from .base_loop import BaseLoop
from .checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                         find_latest_checkpoint, get_state_dict,
                         save_checkpoint, weights_to_cpu)
from .loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from .priority import Priority, get_priority

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]


@RUNNERS.register_module()
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
        work_dir (str): The working directory to save checkpoints. The logs
            will be saved in the subdirectory of `work_dir` named
            :attr:`timestamp`.
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
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_val_loop` for more details.
        test_cfg (dict, optional): A dict to build a test loop. If it does
            not provide "type" key, :class:`TestLoop` will be used by default.
            If ``test_cfg`` specified, :attr:`test_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_test_loop` for more details.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
        optim_wrapper (OptimWrapper or dict, optional):
            Computing gradient of model parameters. If specified,
            :attr:`train_dataloader` should also be specified. If automatic
            mixed precision or gradient accmulation
            training is required. The type of ``optim_wrapper`` should be
            AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
            examples. Defaults to None.
        param_scheduler (_ParamScheduler or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
            See :meth:`build_param_scheduler` for examples.
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
        log_processor (dict, optional): A processor to format logs. Defaults to
            None.
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        visualizer (Visualizer or dict, optional): A Visualizer object or a
            dict build Visualizer object. Defaults to None. If not
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
        >>>     model=dict(type='ToyModel'),
        >>>     work_dir='path/of/work_dir',
        >>>     train_dataloader=dict(
        >>>     dataset=dict(type='ToyDataset'),
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     batch_size=1,
        >>>     num_workers=0),
        >>>     val_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>        batch_size=1,
        >>>        num_workers=0),
        >>>     test_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>         batch_size=1,
        >>>         num_workers=0),
        >>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
        >>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
        >>>         type='SGD', lr=0.01)),
        >>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
        >>>     val_evaluator=dict(type='ToyEvaluator'),
        >>>     test_evaluator=dict(type='ToyEvaluator'),
        >>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
        >>>     val_cfg=dict(),
        >>>     test_cfg=dict(),
        >>>     custom_hooks=[],
        >>>     default_hooks=dict(
        >>>         timer=dict(type='IterTimerHook'),
        >>>         checkpoint=dict(type='CheckpointHook', interval=1),
        >>>         logger=dict(type='LoggerHook'),
        >>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
        >>>         param_scheduler=dict(type='ParamSchedulerHook')),
        >>>     launcher='none',
        >>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
        >>>     log_processor=dict(window_size=20),
        >>>     visualizer=dict(type='Visualizer',
        >>>     vis_backends=[dict(type='LocalVisBackend',
        >>>                        save_dir='temp_dir')])
        >>>    )
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
        >>> runner.test()
    """
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

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
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
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
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optimizer should be either '
                'all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optimizer is None, '
                f'but got {param_scheduler}')

        if param_scheduler is None:
            self.param_schedulers = []
        elif not isinstance(param_scheduler, Sequence):
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
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

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
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        self.default_scope = DefaultScope.get_instance(
            self._experiment_name, scope_name=default_scope)
        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # collect information of all modules registered in the registries
        registries_info = count_registered_modules(
            self.work_dir if self.rank == 0 else None, verbose=False)
        self.logger.debug(registries_info)

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
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)

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
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
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
    def log_dir(self):
        return self._log_dir

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_epochs
        else:
            return 0

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.max_iters
        else:
            return 0

    @property
    def epoch(self):
        """int: Current epoch."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.epoch
        else:
            return 0

    @property
    def iter(self):
        """int: Current iteration."""
        if isinstance(self.train_loop, BaseLoop):
            return self.train_loop.iter
        else:
            return 0

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

    @property
    def train_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._train_loop, BaseLoop) or self._train_loop is None:
            return self._train_loop
        else:
            self._train_loop = self.build_train_loop(self._train_loop)
            return self._train_loop

    @property
    def val_loop(self):
        """:obj:`BaseLoop`: A loop to run validation."""
        if isinstance(self._val_loop, BaseLoop) or self._val_loop is None:
            return self._val_loop
        else:
            self._val_loop = self.build_val_loop(self._val_loop)
            return self._val_loop

    @property
    def test_loop(self):
        """:obj:`BaseLoop`: A loop to run testing."""
        if isinstance(self._test_loop, BaseLoop) or self._test_loop is None:
            return self._test_loop
        else:
            self._test_loop = self.build_test_loop(self._test_loop)
            return self._test_loop

    @property
    def train_dataloader(self):
        """The data loader for training."""
        return self.train_loop.dataloader

    @property
    def val_dataloader(self):
        """The data loader for validation."""
        return self.val_loop.dataloader

    @property
    def test_dataloader(self):
        """The data loader for testing."""
        return self.test_loop.dataloader

    @property
    def val_evaluator(self):
        """:obj:`Evaluator`: An evaluator for validation."""
        return self.val_loop.evaluator

    @property
    def test_evaluator(self):
        """:obj:`Evaluator`: An evaluator for testing."""
        return self.test_loop.evaluator

    @property
    def val_interval(self):
        """int: Interval to run validation during training."""
        return self.train_loop.val_interval

    @property
    def val_begin(self):
        """int: The epoch/iteration to start running validation during
        training."""
        return self.train_loop.val_begin

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
                resource_limit=4096
            )

        Args:
            env_cfg (dict): Config for setting environment.
        """
        if env_cfg.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        mp_cfg: dict = env_cfg.get('mp_cfg', {})
        set_multi_processing(**mp_cfg, distributed=self.distributed)

        # init distributed env first, since logger depends on the dist info.
        if self.distributed:
            dist_cfg: dict = env_cfg.get('dist_cfg', {})
            init_dist(self.launcher, **dist_cfg)

        self._rank, self._world_size = get_dist_info()

        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        # broadcast timestamp from 0 process to other processes
        broadcast(timestamp)
        self._timestamp = time.strftime('%Y%m%d_%H%M%S',
                                        time.localtime(timestamp.item()))

        # https://github.com/pytorch/pytorch/issues/973
        # set resource limit
        if platform.system() != 'Windows':
            import resource
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            base_soft_limit = rlimit[0]
            hard_limit = rlimit[1]
            soft_limit = min(
                max(env_cfg.get('resource_limit', 4096), base_soft_limit),
                hard_limit)
            resource.setrlimit(resource.RLIMIT_NOFILE,
                               (soft_limit, hard_limit))

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
            log_file = osp.join(self._log_dir, f'{self.timestamp}.log')

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

    def build_visualizer(
            self,
            visualizer: Optional[Union[Visualizer,
                                       Dict]] = None) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(
                name=self._experiment_name,
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=self._log_dir)
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            # ensure visualizer containing name key
            visualizer.setdefault('name', self._experiment_name)
            visualizer.setdefault('save_dir', self._log_dir)
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError(
                'visualizer should be Visualizer object, a dict or None, '
                f'but got {visualizer}')

    def build_model(self, model: Union[BaseModel, Dict]) -> BaseModel:
        """Build model.

        If ``model`` is a dict, it will be used to build a nn.Module object
        and initialize the weights if it has ``init_weights`` method.
        Else, if ``model`` is a nn.Module object it will be returned directly.

        An example of ``model``::

            model = dict(type='ResNet')

        Args:
            model (BaseModel or dict): A nn.Module object or a dict to build
                nn.Module object. If ``model`` is a nn.Module object, just
                returns itself.

        Returns:
            nn.Module: Model build from ``model``.
        """
        if isinstance(model, BaseModel):
            return model
        elif isinstance(model, dict):
            model = MODELS.build(model)
            # init weights
            if hasattr(model, 'init_weights'):  # type: ignore
                model.init_weights()  # type: ignore
            return model  # type: ignore
        else:
            raise TypeError('model should be a nn.Module object or dict, '
                            f'but got {model}')

    def wrap_model(
            self, model_wrapper_cfg: Optional[Dict],
            model: BaseModel) -> Union[DistributedDataParallel, BaseModel]:
        """Wrap the model to :obj:``MMDistributedDataParallel`` or other custom
        distributed data-parallel module wrappers.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``DistributedDataParallel`` will be used in
                distributed environment. Defaults to None.
            model (BaseModel): Model to be wrapped.

        Returns:
            BaseModel or DistributedDataParallel: BaseModel or subclass of
            ``DistributedDataParallel``.
        """
        if is_model_wrapper(model):
            if model_wrapper_cfg is not None:
                raise TypeError(
                    'model has been wrapped and "model_wrapper_cfg" should be '
                    f'None, but got {model_wrapper_cfg}')

            return model

        # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
        model = model.to(get_device())

        if not self.distributed:
            self.logger.info(
                'Distributed training is not used, all SyncBatchNorm (SyncBN) '
                'layers in the model will be automatically reverted to '
                'BatchNormXd layers if they are used.')
            model = revert_sync_batchnorm(model)
            return model

        if model_wrapper_cfg is None:
            find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                  False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            # TODO: may use a more elegant way to get local device ID.
            model = MMDistributedDataParallel(
                module=model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MODEL_WRAPPERS.build(
                model_wrapper_cfg, default_args=dict(module=model))
        return model

    def scale_lr(self,
                 optim_wrapper: OptimWrapper,
                 auto_scale_lr: Optional[Dict] = None) -> None:
        """Automatically scaling learning rate in training according to the
        ratio of ``base_batch_size`` in ``autoscalelr_cfg`` and real batch
        size.

        It scales the learning rate linearly according to the
        `paper <https://arxiv.org/abs/1706.02677>`_.

        Note:
            ``scale_lr`` must be called after building optimizer wrappers
            and before building parameter schedulers.

        Args:
            optim_wrapper (OptimWrapper): An OptimWrapper object whose
                parameter groups' learning rate need to be scaled.
            auto_scale_lr (Dict, Optional): Config to scale the learning
                rate automatically. It includes ``base_batch_size`` and
                ``enable``. ``base_batch_size`` is the batch size that the
                optimizer lr is based on. ``enable`` is the switch to turn on
                and off the feature.
        """
        if (auto_scale_lr is None or not auto_scale_lr.get('enable', False)):
            return None

        assert 'base_batch_size' in auto_scale_lr, \
            'Lack of `base_batch_size` in `auto_scale_lr`.'
        dataloader: Union[DataLoader, Dict] = self._train_dataloader
        bs = dataloader.batch_size if isinstance(
            dataloader, DataLoader) else dataloader['batch_size']
        real_bs = self.world_size * bs
        base_bs = auto_scale_lr['base_batch_size']
        ratio = float(real_bs) / float(base_bs)
        self.logger.info(f'LR is set based on batch size of {base_bs} '
                         f'and the current batch size is {real_bs}. '
                         f'Scaling the original LR by {ratio}.')

        def _is_built(schedulers):
            if isinstance(schedulers, dict):
                return False if 'type' in schedulers else any(
                    _is_built(s) for s in schedulers.values())
            if isinstance(schedulers, list):
                return any(_is_built(s) for s in schedulers)
            return isinstance(schedulers, _ParamScheduler)

        if _is_built(self.param_schedulers):
            raise RuntimeError('`scale_lr` should be called before building '
                               'ParamScheduler because ParamScheduler will '
                               'store initial lr from optimizer wrappers')

        assert isinstance(optim_wrapper, OptimWrapper), \
            '`scale_lr should be called after building OptimWrapper'
        wrappers = list(optim_wrapper.values()) if isinstance(
            optim_wrapper, OptimWrapperDict) else [optim_wrapper]
        for wrapper in wrappers:
            for group in wrapper.optimizer.param_groups:
                group['lr'] = group['lr'] * ratio

    def build_optim_wrapper(
        self, optim_wrapper: Union[Optimizer, OptimWrapper, Dict]
    ) -> Union[OptimWrapper, OptimWrapperDict]:
        """Build optimizer wrapper.

        If ``optim_wrapper`` is a config dict for only one optimizer,
        the keys must contain ``optimizer``, and ``type`` is optional.
        It will build a :obj:`OptimWrapper` by default.

        If ``optim_wrapper`` is a config dict for multiple optimizers, i.e.,
        it has multiple keys and each key is for an optimizer wrapper. The
        constructor must be specified since
        :obj:`DefaultOptimizerConstructor` cannot handle the building of
        training with multiple optimizers.

        If ``optim_wrapper`` is a dict of pre-built optimizer wrappers, i.e.,
        each value of ``optim_wrapper`` represents an ``OptimWrapper``
        instance. ``build_optim_wrapper`` will directly build the
        :obj:`OptimWrapperDict` instance from ``optim_wrapper``.

        Args:
            optim_wrapper (OptimWrapper or dict): An OptimWrapper object or a
                dict to build OptimWrapper objects. If ``optim_wrapper`` is an
                OptimWrapper, just return an ``OptimizeWrapper`` instance.

        Note:
            For single optimizer training, if `optim_wrapper` is a config
            dict, `type` is optional(defaults to :obj:`OptimWrapper`) and it
            must contain `optimizer` to build the corresponding optimizer.

        Examples:
            >>> # build an optimizer
            >>> optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=dict(
            ...     type='SGD', lr=0.01))
            >>> # optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> # is also valid.
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build optimizer without `type`
            >>> optim_wrapper_cfg = dict(optimizer=dict(type='SGD', lr=0.01))
            >>> optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.01
                maximize: False
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            >>> # build multiple optimizers
            >>> optim_wrapper_cfg = dict(
            ...    generator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='SGD', lr=0.01)),
            ...    discriminator=dict(type='OptimWrapper', optimizer=dict(
            ...        type='Adam', lr=0.001))
            ...    # need to customize a multiple optimizer constructor
            ...    constructor='CustomMultiOptimizerConstructor',
            ...)
            >>> optim_wrapper = runner.optim_wrapper(optim_wrapper_cfg)
            >>> optim_wrapper
            name: generator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            SGD (
            Parameter Group 0
                dampening: 0
                lr: 0.1
                momentum: 0
                nesterov: False
                weight_decay: 0
            )
            name: discriminator
            Type: OptimWrapper
            accumulative_counts: 1
            optimizer:
            'discriminator': Adam (
            Parameter Group 0
                dampening: 0
                lr: 0.02
                momentum: 0
                nesterov: False
                weight_decay: 0
            )

        Important:
            If you need to build multiple optimizers, you should implement a
            MultiOptimWrapperConstructor which gets parameters passed to
            corresponding optimizers and compose the ``OptimWrapperDict``.
            More details about how to customize OptimizerConstructor can be
            found at `optimizer-docs`_.

        Returns:
            OptimWrapper: Optimizer wrapper build from ``optimizer_cfg``.

        .. _optimizer-docs:
           https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
        """
        if isinstance(optim_wrapper, OptimWrapper):
            return optim_wrapper
        elif isinstance(optim_wrapper, (dict, ConfigDict, Config)):
            # If `optim_wrapper` is a config dict with only one optimizer,
            # the config dict must contain `optimizer`:
            # optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.1))
            # `type` is optional, defaults to `OptimWrapper`.
            # `optim_wrapper` could also be defined as:
            # optim_wrapper = dict(type='AmpOptimWrapper', optimizer=dict(type='SGD', lr=0.1))  # noqa: E501
            # to build specific optimizer wrapper.
            if 'type' in optim_wrapper or 'optimizer' in optim_wrapper:
                optim_wrapper = build_optim_wrapper(self.model, optim_wrapper)
                return optim_wrapper
            elif 'constructor' not in optim_wrapper:
                # if `type` and `optimizer` are not defined in `optim_wrapper`,
                # it should be the case of training with multiple optimizers.
                # If constructor is not defined in `optim_wrapper`, each value
                # of `optim_wrapper` must be an `OptimWrapper` instance since
                # `DefaultOptimizerConstructor` will not handle the case of
                # training with multiple optimizers. `build_optim_wrapper` will
                # directly build the `OptimWrapperDict` instance from
                # `optim_wrapper.`
                optim_wrappers = OrderedDict()
                for name, optim in optim_wrapper.items():
                    if not isinstance(optim, OptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"type" and "constructor" are not in '
                            f'optimizer, but got {name}={optim}')
                    optim_wrappers[name] = optim
                return OptimWrapperDict(**optim_wrappers)
                # If constructor is defined, directly build the optimizer
                # wrapper instance from the config dict.
            else:
                optim_wrapper = build_optim_wrapper(self.model, optim_wrapper)
                return optim_wrapper
        else:
            raise TypeError('optimizer wrapper should be an OptimWrapper '
                            f'object or dict, but got {optim_wrapper}')

    def _build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict, List],
            optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.
            optim_wrapper (OptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.
        """
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)
                convert_to_iter = _scheduler.pop('convert_to_iter_based',
                                                 False)
                if convert_to_iter:
                    assert _scheduler.get(
                        'by_epoch',
                        True), ('only epoch-based parameter scheduler can be '
                                'converted to iter-based')
                    assert isinstance(self._train_loop, BaseLoop), \
                        'Scheduler can only be converted to iter-based ' \
                        'when train loop is built.'
                    cls = PARAM_SCHEDULERS.get(_scheduler.pop('type'))
                    param_schedulers.append(
                        cls.build_iter_from_epoch(  # type: ignore
                            optimizer=optim_wrapper,
                            **_scheduler,
                            epoch_length=len(
                                self.train_dataloader),  # type: ignore
                        ))
                else:
                    param_schedulers.append(
                        PARAM_SCHEDULERS.build(
                            _scheduler,
                            default_args=dict(optimizer=optim_wrapper)))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')

        return param_schedulers

    def build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict,
                                   List]) -> ParamSchedulerType:
        """Build parameter schedulers.

        ``build_param_scheduler`` should be called after
        ``build_optim_wrapper`` because the building logic will change
        according to the number of optimizers built by the runner.
        The cases are as below:

        - Single optimizer: When only one optimizer is built and used in the
          runner, ``build_param_scheduler`` will return a list of
          parameter schedulers.
        - Multiple optimizers: When two or more optimizers are built and used
          in runner, ``build_param_scheduler`` will return a dict containing
          the same keys with multiple optimizers and each value is a list of
          parameter schedulers. Note that, if you want different optimizers to
          use different parameter shedulers to update optimizer's
          hyper-parameters, the input parameter ``scheduler`` also needs to be
          a dict and its key are consistent with multiple optimizers.
          Otherwise, the same parameter schedulers will be used to update
          optimizer's hyper-parameters.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.

        Examples:
            >>> # build one scheduler
            >>> optim_cfg = dict(dict(type='SGD', lr=0.01))
            >>> runner.optim_wrapper = runner.build_optim_wrapper(
            >>>     optim_cfg)
            >>> scheduler_cfg = dict(type='MultiStepLR', milestones=[1, 2])
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f6966290>]  # noqa: E501

            >>> # build multiple schedulers
            >>> scheduler_cfg = [
            ...    dict(type='MultiStepLR', milestones=[1, 2]),
            ...    dict(type='StepLR', step_size=1)
            ... ]
            >>> schedulers = runner.build_param_scheduler(scheduler_cfg)
            >>> schedulers
            [<mmengine.optim.scheduler.lr_scheduler.MultiStepLR at 0x7f70f60dd3d0>,  # noqa: E501
            <mmengine.optim.scheduler.lr_scheduler.StepLR at 0x7f70f6eb6150>]

        Above examples only provide the case of one optimizer and one scheduler
        or multiple shedulers. If you want to know how to set parameter
        scheduler when using multiple optimizers, you can find more examples
        `optimizer-docs`_.

        Returns:
            list[_ParamScheduler] or dict[str, list[_ParamScheduler]]: List of
            parameter schedulers or a dictionary contains list of parameter
            schedulers build from ``scheduler``.

        .. _optimizer-docs:
           https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
        """
        param_schedulers: ParamSchedulerType
        if not isinstance(self.optim_wrapper, OptimWrapperDict):
            # Since `OptimWrapperDict` inherits from `OptimWrapper`,
            # `isinstance(self.optim_wrapper, OptimWrapper)` cannot tell
            # whether `self.optim_wrapper` is an `OptimizerWrapper` or
            # `OptimWrapperDict` instance. Therefore, here we simply check
            # self.optim_wrapper is not an `OptimWrapperDict` instance and
            # then assert it is an OptimWrapper instance.
            assert isinstance(self.optim_wrapper, OptimWrapper), (
                '`build_optimizer` should be called before'
                '`build_param_scheduler` because the latter depends '
                'on the former')
            param_schedulers = self._build_param_scheduler(
                scheduler, self.optim_wrapper)  # type: ignore
            return param_schedulers
        else:
            param_schedulers = dict()
            for name, optimizer in self.optim_wrapper.items():
                if isinstance(scheduler, dict) and 'type' not in scheduler:
                    # scheduler is a dict and each item is a ParamScheduler
                    # object or a config to build ParamScheduler objects
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler[name], optimizer)
                else:
                    param_schedulers[name] = self._build_param_scheduler(
                        scheduler, optimizer)

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
            evaluator (Evaluator or dict or list): An Evaluator object or a
                config dict or list of config dict used to build an Evaluator.

        Returns:
            Evaluator: Evaluator build from ``evaluator``.
        """
        if isinstance(evaluator, Evaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            # if `metrics` in dict keys, it means to build customized evalutor
            if 'metrics' in evaluator:
                assert 'type' in evaluator, 'expected customized evaluator' \
                                    f' with key `type`, but got {evaluator}'
                return EVALUATOR.build(evaluator)
            # otherwise, default evalutor will be built
            else:
                return Evaluator(evaluator)  # type: ignore
        elif is_list_of(evaluator, dict):
            # use the default `Evaluator`
            return Evaluator(evaluator)  # type: ignore
        else:
            raise TypeError(
                'evaluator should be one of dict, list of dict, and Evaluator'
                f', but got {evaluator}')

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None) -> DataLoader:
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
            seed (int, optional): Random seed. Defaults to None.

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
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler = DATA_SAMPLERS.build(
                sampler_cfg, default_args=dict(dataset=dataset, seed=seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]
        if seed is not None:
            init_fn = partial(
                worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=get_rank(),
                seed=seed)
        else:
            init_fn = None

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, to make this more flexible, collate_fn in MMengine does
        # nothing. The action to merge a list of samples will be handled
        # in model.
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
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
                    runner=self, dataloader=self._train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
        return loop  # type: ignore

    def build_val_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build validation loop.

        Examples of ``loop``:

            # `ValLoop` will be used
            loop = dict()

            # custom validation loop
            loop = dict(type='CustomValLoop')

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
                    dataloader=self._val_dataloader,
                    evaluator=self._val_evaluator))
        else:
            loop = ValLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._val_dataloader,
                evaluator=self._val_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test loop.

        Examples of ``loop``::

            # `TestLoop` will be used
            loop = dict()

            # custom test loop
            loop = dict(type='CustomTestLoop')

        Args:
            loop (BaseLoop or dict): A test loop or a dict to build test loop.
                If ``loop`` is a test loop object, just returns itself.

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
                    dataloader=self._test_dataloader,
                    evaluator=self._test_evaluator))
        else:
            loop = TestLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._test_dataloader,
                evaluator=self._test_evaluator)  # type: ignore

        return loop  # type: ignore

    def build_log_processor(
            self, log_processor: Union[LogProcessor, Dict]) -> LogProcessor:
        """Build test log_processor.

        Examples of ``log_processor``:

            # `LogProcessor` will be used
            log_processor = dict()

            # custom log_processor
            log_processor = dict(type='CustomLogProcessor')

        Args:
            log_processor (LogProcessor or dict): A log processor or a dict
            to build log processor. If ``log_processor`` is a log processor
            object, just returns itself.

        Returns:
            :obj:`LogProcessor`: Log processor object build from
            ``log_processor_cfg``.
        """
        if isinstance(log_processor, LogProcessor):
            return log_processor
        elif not isinstance(log_processor, dict):
            raise TypeError(
                'log processor should be a LogProcessor object or dict, but'
                f'got {log_processor}')

        log_processor_cfg = copy.deepcopy(log_processor)  # type: ignore

        if 'type' in log_processor_cfg:
            log_processor = LOG_PROCESSORS.build(log_processor_cfg)
        else:
            log_processor = LogProcessor(**log_processor_cfg)  # type: ignore

        return log_processor  # type: ignore

    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True

    def train(self) -> None:
        """Launch training."""
        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore

        self.call_hook('before_run')
        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # TODO: add a contextmanager to avoid calling `before_run` many times
        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        self.train_loop.run()  # type: ignore
        self.call_hook('after_run')

    def val(self) -> None:
        """Launch validation."""
        if self._val_loop is None:
            raise RuntimeError(
                '`self._val_loop` should not be None when calling val method.'
                'Please provide `val_dataloader`, `val_cfg` and '
                '`val_evaluator` arguments when initializing runner.')

        self._val_loop = self.build_val_loop(self._val_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        self.val_loop.run()  # type: ignore
        self.call_hook('after_run')

    def test(self) -> None:
        """Launch test."""
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

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
                try:
                    getattr(hook, fn_name)(self, **kwargs)
                except TypeError as e:
                    raise TypeError(f'{e} in {hook}') from None

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
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
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
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                sampler_seed=dict(type='DistSamplerSeedHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
            )

        If not None, ``hooks`` will be merged into ``default_hooks``.
        If there are None value in default_hooks, the corresponding item will
        be popped from ``default_hooks``::

            hooks = dict(timer=None)

        The final registered default hooks will be :obj:`RuntimeInfoHook`,
        :obj:`DistSamplerSeedHook`, :obj:`LoggerHook`,
        :obj:`ParamSchedulerHook` and :obj:`CheckpointHook`.

        Args:
            hooks (dict[str, Hook or dict], optional): Default hooks or configs
                to be registered.
        """
        default_hooks: dict = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            sampler_seed=dict(type='DistSamplerSeedHook'),
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
                device = get_device()
                checkpoint = self.load_checkpoint(
                    filename,
                    map_location=lambda storage, loc: storage.to(device))
            else:
                checkpoint = self.load_checkpoint(filename)
        else:
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location)

        self.train_loop._epoch = checkpoint['meta']['epoch']
        self.train_loop._iter = checkpoint['meta']['iter']

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
                if (self.auto_scale_lr is None
                        or not self.auto_scale_lr.get('enable', False)):
                    raise RuntimeError(
                        'Cannot automatically rescale lr in resuming. Please '
                        'make sure the number of GPU is consistent with the '
                        'previous training state resuming from the checkpoint '
                        'or set `enable` in `auto_scale_lr to False.')

        # resume random seed
        resumed_seed = checkpoint['meta'].get('seed', None)
        current_seed = self._randomness_cfg.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                warnings.warn(f'The value of random seed in the '
                              f'checkpoint "{resumed_seed}" is '
                              f'different from the value in '
                              f'`randomness` config "{current_seed}"')
            self._randomness_cfg.update(seed=resumed_seed)
            self.set_randomness(**self._randomness_cfg)

        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        if (dataset_meta is not None
                and dataset_meta != self.train_dataloader.dataset.metainfo):
            warnings.warn(
                'The dataset metainfo from the resumed checkpoint is '
                'different from the current training dataset, please '
                'check the correctness of the checkpoint or the training '
                'dataset.')

        self.message_hub = checkpoint['message_hub']

        # resume optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
            self.optim_wrapper.load_state_dict(  # type: ignore
                checkpoint['optimizer'])

        # resume param scheduler
        if 'param_schedulers' in checkpoint and resume_param_scheduler:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)
            if isinstance(self.param_schedulers, dict):
                for name, schedulers in self.param_schedulers.items():
                    for scheduler, ckpt_scheduler in zip(
                            schedulers, checkpoint['param_schedulers'][name]):
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(
                        self.param_schedulers, checkpoint['param_schedulers']):
                    scheduler.load_state_dict(ckpt_scheduler)  # type: ignore

        self._has_loaded = True

        self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

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
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint

    @master_only
    def save_checkpoint(self,
                        out_dir: str,
                        filename: str,
                        file_client_args: Optional[dict] = None,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        meta: dict = None,
                        by_epoch: bool = True):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Default: None.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Whether the scheduled momentum is updated by
                epochs. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch, iter=self.iter + 1)

        file_client = FileClient.infer_client(file_client_args, out_dir)
        filepath = file_client.join_path(out_dir, filename)

        meta.update(
            cfg=self.cfg.pretty_text,
            dataset_meta=self.train_dataloader.dataset.metainfo,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model)),
            'message_hub': self.message_hub
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = self.optim_wrapper.state_dict()
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(checkpoint, filepath)

        save_file = osp.join(self.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(filepath)

    @master_only
    def dump_config(self) -> None:
        """Dump config to `work_dir`."""
        if self.cfg.filename is not None:
            filename = osp.basename(self.cfg.filename)
        else:
            filename = f'{self.timestamp}.py'
        self.cfg.dump(osp.join(self.work_dir, filename))
