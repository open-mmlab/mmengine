import argparse
import os

from mmengine.config import Config
from mmengine.strategy import DDPStrategy, NativeStrategy

from mmcls.utils import register_all_modules
register_all_modules(init_default_scope=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument(
        '--ddp', action='store_true', help='enable ddp training')
    parser.add_argument(
        '--scope', type=str, help='default scope', default='mmengine')
    return parser.parse_args()


def detect_launcher():
    if os.getenv('RANK') is not None:
        return 'pytorch'
    if os.getenv('OMPI_COMM_WORLD_LOCAL_RANK') is not None:
        return 'mpi'
    if os.getenv('SLURM_PROCID') is not None:
        return 'slurm'
    return 'none'


if __name__ == '__main__':
    args = get_args()
    config = Config.fromfile(args.config)

    if args.ddp:
        strategy = DDPStrategy(amp=True)
    else:
        strategy = NativeStrategy()

    strategy.setup_distributed(launcher=detect_launcher())
    model, optim_wrapper, param_schedulers = strategy.setup(
        config.model, config.optim_wrapper, config.param_scheduler, cfg=config)
    breakpoint()
