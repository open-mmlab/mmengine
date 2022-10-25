# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmengine.runner import Runner

from mmengine.hooks import Hook
from mmengine.registry import HOOKS, MODELS


@HOOKS.register_module()
class PrepareTTAHook(Hook):
    """Wraps `runner.model` with subclass of :class:`BaseTTAModel` in
    `before_test`.

    Args:
        tta_cfg (dict): Config dictionary of the test time augmentation model.
    """

    def __init__(self, tta_cfg: dict):
        self.tta_cfg = tta_cfg

    def before_test(self, runner: 'Runner') -> None:
        """Wraps `runner.model` with the subclass of :class:`BaseTTAModel`.

        Args:
            runner (Runner): The runner of the testing process.
        """
        self.tta_cfg['module'] = runner.model  # type: ignore
        model_wrapper = MODELS.build(self.tta_cfg)
        runner.model = model_wrapper  # type: ignore
