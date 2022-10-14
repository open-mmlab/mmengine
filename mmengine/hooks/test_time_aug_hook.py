# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict
from mmengine.hooks import Hook
from mmengine.registry import MODEL_WRAPPERS


class PrepareTTAHook(Hook):

    def __init__(self, tta_cfg: ConfigDict):
        self.tta_cfg = tta_cfg

    def before_test(self, runner) -> None:
        self.tta_cfg.module = runner.model
        model_wrapper = MODEL_WRAPPERS.build(self.tta_cfg)
        runner.model = model_wrapper
