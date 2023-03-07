from copy import deepcopy

from mmengine.hub import get_config
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.runner import Runner


@MODELS.register_module()
class ClsTTAModel(BaseTTAModel):

    def merge_preds(self, data_samples_list):
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(self, data_samples):
        merged_data_sample = data_samples[0].new()
        merged_score = sum(data_sample.pred_label.score
                           for data_sample in data_samples) / len(data_samples)
        merged_data_sample.set_pred_score(merged_score)
        return merged_data_sample


if __name__ == '__main__':
    cfg = get_config('mmcls::resnet/resnet50_8xb16_cifar10.py')
    cfg.work_dir = 'work_dirs/resnet50_8xb16_cifar10'
    cfg.model = dict(type='ClsTTAModel', module=cfg.model)
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    flip_tta = dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='RandomFlip', prob=1.),
                dict(type='RandomFlip', prob=0.)
            ],
            [test_pipeline[-1]],
        ])
    # Replace the last transform with `TestTimeAug`
    cfg.test_dataloader.dataset.pipeline[-1] = flip_tta
    cfg.load_from = 'https://download.openmmlab.com/mmclassification/v0' \
                    '/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth'
    runner = Runner.from_cfg(cfg)
    runner.test()
