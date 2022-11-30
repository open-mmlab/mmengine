# Copyright (c) OpenMMLab. All rights reserved.
base1 = 'base1'
_base_ = [
    f'./{base1}.py', '../yaml_config/base2.yaml', '../json_config/base3.json',
    './base4.py'
]

item3 = False
item4 = 'test'
item8 = '{{fileBasename}}'
item9 = {{_base_.item2}}
item10 = {{_base_.item7.b.c}}
