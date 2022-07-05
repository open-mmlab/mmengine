# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    './base1.py', '../yaml_config/base2.yaml', '../json_config/base3.json',
    './base4.py'
]
item2 = dict(b=[5, 6])
item3 = False
item4 = 'test'
_base_.item6[0] = dict(c=0)
item8 = '{{fileBasename}}'
item9, item10, item11 = _base_.item7['b']['c']
