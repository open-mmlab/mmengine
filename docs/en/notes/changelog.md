# Changelog of v0.x

## v0.8.2 (07/12/2023)

### Bug fixes

- Fix pickling the Python style config by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1241
- Fix the logic of setting `lazy_import` by [@Li-Qingyun](https://github.com/Li-Qingyun) in https://github.com/open-mmlab/mmengine/pull/1239

## v0.8.1 (07/05/2023)

### New Features & Enhancements

- Accelerate `Config.dump` and support converting Lazyxxx to string in `ConfigDict.to_dict`by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1232

### Bug fixes

- FSDP should call `_get_ignored_modules` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1235

### Docs

- Add a document to introduce how to train a large model by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1228

## v0.8.0 (06/30/2023)

### Highlights

- Support training with [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=fsdp) and [DeepSpeed](https://www.deepspeed.ai/). Refer to the [example](https://github.com/open-mmlab/mmengine/blob/main/examples/distributed_training_with_flexible_runner.py) for more detailed usages.

- Introduce the pure Python style configuration file:

  - Support navigating to base configuration file in IDE
  - Support navigating to base variable in IDE
  - Support navigating to source code of class in IDE
  - Support inheriting two configuration files containing the same field
  - Load the configuration file without other third-party requirements

  Refer to the [tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) for more detailed usages.

  ![new-config-en](https://github.com/open-mmlab/mmengine/assets/57566630/7eb41748-9374-488f-901e-fcd7f0d3c8a1)

### New Features & Enhancements

- Support training with FSDP by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1213
- Add `FlexibleRunner` and `Strategies`, and support training with DeepSpeed by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1183
- Support pure Python style configuration file by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1071
- Learning rate in log can show the base learning rate of optimizer by [@AkideLiu](https://github.com/AkideLiu) in https://github.com/open-mmlab/mmengine/pull/1019
- Refine the error message when auto_scale_lr is not set correctly by [@alexander-soare](https://github.com/alexander-soare) in https://github.com/open-mmlab/mmengine/pull/1181
- WandbVisBackend supports updating config by [@zgzhengSEU](https://github.com/zgzhengSEU) in https://github.com/open-mmlab/mmengine/pull/977

### Bug fixes

- CheckpointHook should check whether file exists before removing it by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1198
- Fix undefined variable error in Runner by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1219

### Docs

- Add a document to introduce how to debug with vscode by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1212
- Update English introduction by [@evdcush](https://github.com/evdcush) in https://github.com/open-mmlab/mmengine/pull/1189
- Fix parameter typing error in document by [@syo093c](https://github.com/syo093c) in https://github.com/open-mmlab/mmengine/pull/1201
- Fix gpu collection during evaluation by [@edkair](https://github.com/edkair) in https://github.com/open-mmlab/mmengine/pull/1208
- Fix a comment in runner tutorial by [@joihn](https://github.com/joihn) in https://github.com/open-mmlab/mmengine/pull/1210

### Contributors

A total of 9 developers contributed to this release. Thanks [@evdcush](https://github.com/evdcush), [@zhouzaida](https://github.com/zhouzaida), [@AkideLiu](https://github.com/AkideLiu), [@joihn](https://github.com/joihn), [@HAOCHENYE](https://github.com/HAOCHENYE), [@edkair](https://github.com/edkair), [@alexander-soare](https://github.com/alexander-soare), [@syo093c](https://github.com/syo093c), [@zgzhengSEU](https://github.com/zgzhengSEU)

## v0.7.4 (06/03/2023)

### Highlights

- Support using `ClearML` to record experiment data
- Add `Sophia` optimizers

### New Features & Enhancements

- Add visualize backend for clearml by [@gachiemchiep](https://github.com/gachiemchiep) in https://github.com/open-mmlab/mmengine/pull/1091
- Support Sophia optimizers by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1170
- Refactor unittest syncbuffer by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/813
- Allow `ann_file`, `data_root` is `None` for `BaseDataset` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/850
- Enable full precision training on Ascend NPU by [@Ginray](https://github.com/Ginray) in https://github.com/open-mmlab/mmengine/pull/1109
- Creating a text classification example by [@TankNee](https://github.com/TankNee) in https://github.com/open-mmlab/mmengine/pull/1122
- Add option to log selected config only by [@KickCellarDoor](https://github.com/KickCellarDoor) in https://github.com/open-mmlab/mmengine/pull/1159
- Add an option to control whether to show progress bar in BaseInference by [@W-ZN](https://github.com/W-ZN) in https://github.com/open-mmlab/mmengine/pull/1135
- Support dipu device by [@CokeDong](https://github.com/CokeDong) in https://github.com/open-mmlab/mmengine/pull/1127
- Let unit tests not affect each other by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1169
- Add support for full wandb's `define_metric` arguments by [@i-aki-y](https://github.com/i-aki-y) in https://github.com/open-mmlab/mmengine/pull/1099

### Bug fixes

- Fix the incorrect device of inputs in get_model_complexity_info by [@CescMessi](https://github.com/CescMessi) in https://github.com/open-mmlab/mmengine/pull/1130
- Correctly saves `_metadata` of `state_dict` when saving checkpoints by [@Bomsw](https://github.com/Bomsw) in https://github.com/open-mmlab/mmengine/pull/1131
- Correctly record random seed in log by [@Shiyang980713](https://github.com/Shiyang980713) in https://github.com/open-mmlab/mmengine/pull/1152
- Close MLflowVisBackend only if active by [@zimonitrome](https://github.com/zimonitrome) in https://github.com/open-mmlab/mmengine/pull/1151
- Fix `ProfileHook` cannot profile ddp-training by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1140
- Handle the case for Multi-Instance GPUs when using `cuda_visible_devices` by [@adrianjoshua-strutt](https://github.com/adrianjoshua-strutt) in https://github.com/open-mmlab/mmengine/pull/1164
- Fix attribute error when parsing `CUDA_VISIBLE_DEVICES` in logger [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/1172

### Docs

- Translate `infer.md` by [@Hongru-Xiao](https://github.com/Hongru-Xiao) in https://github.com/open-mmlab/mmengine/pull/1121
- Fix a missing comma in `tutorials/runner.md` by [@gy-7](https://github.com/gy-7) in https://github.com/open-mmlab/mmengine/pull/1146
- Fix typo in comment by [@YQisme](https://github.com/YQisme) in https://github.com/open-mmlab/mmengine/pull/1154
- Translate `data_element.md` by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/1067
- Add the usage of clearml by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1180

### Contributors

A total of 19 developers contributed to this release. Thanks [@Hongru-Xiao](https://github.com/Hongru-Xiao)  [@i-aki-y](https://github.com/i-aki-y)  [@Bomsw](https://github.com/Bomsw)  [@KickCellarDoor](https://github.com/KickCellarDoor)  [@zhouzaida](https://github.com/zhouzaida)  [@YQisme](https://github.com/YQisme)  [@gachiemchiep](https://github.com/gachiemchiep)  [@CescMessi](https://github.com/CescMessi)  [@W-ZN](https://github.com/W-ZN)  [@Ginray](https://github.com/Ginray)  [@adrianjoshua-strutt](https://github.com/adrianjoshua-strutt)  [@CokeDong](https://github.com/CokeDong)  [@xin-li-67](https://github.com/xin-li-67)  [@Xiangxu-0103](https://github.com/Xiangxu-0103)  [@HAOCHENYE](https://github.com/HAOCHENYE)  [@Shiyang980713](https://github.com/Shiyang980713)  [@TankNee](https://github.com/TankNee)  [@zimonitrome](https://github.com/zimonitrome)  [@gy-7](https://github.com/gy-7)

## v0.7.3 (04/28/2023)

### Highlights

- Support using MLflow to record experiment data
- Support registering callable objects to the registry

### New Features & Enhancements

- Add `MLflowVisBackend` by [@sh0622-kim](https://github.com/sh0622-kim) in https://github.com/open-mmlab/mmengine/pull/878
- Support customizing `worker_init_fn` in dataloader config by [@shufanwu](https://github.com/shufanwu) in https://github.com/open-mmlab/mmengine/pull/1038
- Make the parameters of get_model_complexity_info() friendly by [@sjiang95](https://github.com/sjiang95) in https://github.com/open-mmlab/mmengine/pull/1056
- Add torch_npu optimizer by [@luomaoling](https://github.com/luomaoling) in https://github.com/open-mmlab/mmengine/pull/1079
- Support registering callable objects [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/595
- Complement type hint of get_model_complexity_info() by [@sjiang95](https://github.com/sjiang95) in https://github.com/open-mmlab/mmengine/pull/1064
- MessageHub.get_info() supports returning a default value by [@enkilee](https://github.com/enkilee) in https://github.com/open-mmlab/mmengine/pull/991
- Refactor logger hook unit test by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/797
- Support BoolTensor and LongTensor on Ascend NPU by [@Ginray](https://github.com/Ginray) in https://github.com/open-mmlab/mmengine/pull/1011
- Remove useless variable declaration by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1052
- Enhance the support for MLU device by [@josh6688](https://github.com/josh6688) in https://github.com/open-mmlab/mmengine/pull/1075
- Support configuring synchronization directory for BaseMetric by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1074
- Support accepting multiple `input_shape` for `get_model_complexity_info` by [@sjiang95](https://github.com/sjiang95) in https://github.com/open-mmlab/mmengine/pull/1065
- Enhance docstring and error catching in `MessageHub`  by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1098
- Enhance the efficiency of Visualizer.show by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1015
- Update repo list by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1108
- Enhance error message during custom import by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1102
- Support `_load_state_dict_post_hooks` in `load_state_dict` by [@mzr1996](https://github.com/mzr1996) in https://github.com/open-mmlab/mmengine/pull/1103

### Bug fixes

- Fix publishing multiple checkpoints when using multiple GPUs by [@JunweiZheng93](https://github.com/JunweiZheng93) in https://github.com/open-mmlab/mmengine/pull/1070
- Fix error when `log_with_hierarchy` is `True` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1085
- Call SyncBufferHook before validation in IterBasedTrainLoop by [@Luo-Yihang](https://github.com/Luo-Yihang) in https://github.com/open-mmlab/mmengine/pull/982
- Fix the resuming error caused by HistoryBuffer by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1078
- Failed to remove the previous best checkpoints by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1086
- Fix using incorrect local rank by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/973
- No training log when the num of iterations is smaller than the default interval by [@shufanwu](https://github.com/shufanwu) in https://github.com/open-mmlab/mmengine/pull/1046
- `collate_fn` could not be a function object by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1093
- Fix `optimizer.state` could be saved in cuda:0 by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/966
- Fix building unnecessary loop during train/test/val by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1107

### Docs

- Introduce the use of wandb and tensorboard by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/912
- Translate tutorials/evaluation.md by [@LEFTeyex](https://github.com/LEFTeyex) in https://github.com/open-mmlab/mmengine/pull/1053
- Translate design/evaluation.md by [@zccjjj](https://github.com/zccjjj) in https://github.com/open-mmlab/mmengine/pull/1062
- Fix three typos in runner by [@jsrdcht](https://github.com/jsrdcht) in https://github.com/open-mmlab/mmengine/pull/1068
- Translate migration/hook.md to English by [@SheffieldCao](https://github.com/SheffieldCao) in https://github.com/open-mmlab/mmengine/pull/1054
- Replace MMCls with MMPretrain in docs by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1096

### Contributors

A total of 17 developers contributed to this release. Thanks [@enkilee](https://github.com/enkilee), [@JunweiZheng93](https://github.com/JunweiZheng93), [@sh0622-kim](https://github.com/sh0622-kim), [@jsrdcht](https://github.com/jsrdcht), [@SheffieldCao](https://github.com/SheffieldCao), [@josh6688](https://github.com/josh6688), [@mzr1996](https://github.com/mzr1996), [@zhouzaida](https://github.com/zhouzaida), [@shufanwu](https://github.com/shufanwu), [@Luo-Yihang](https://github.com/Luo-Yihang), [@C1rN09](https://github.com/C1rN09), [@LEFTeyex](https://github.com/LEFTeyex), [@zccjjj](https://github.com/zccjjj), [@Ginray](https://github.com/Ginray), [@HAOCHENYE](https://github.com/HAOCHENYE), [@sjiang95](https://github.com/sjiang95), [@luomaoling](https://github.com/luomaoling)

## v0.7.2 (04/06/2023)

### Bug fixes

- Align the evaluation result in log by [@kitecats](https://github.com/kitecats) in https://github.com/open-mmlab/mmengine/pull/1034
- Update the logic to calculate the `repeat_factors` in `ClassBalancedDataset` by [@BIGWangYuDong](https://github.com/BIGWangYuDong) in https://github.com/open-mmlab/mmengine/pull/1048
- Initialize sub-modules in `DistributedDataParallel` that define `init_weights` during initialization by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1045
- Refactor checkpointhook unittest by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/789

### Contributors

A total of 3 developers contributed to this release. Thanks [@kitecats](https://github.com/kitecats), [@BIGWangYuDong](https://github.com/BIGWangYuDong), [@HAOCHENYE](https://github.com/HAOCHENYE)

## v0.7.1 (04/03/2023)

### Highlights

- Support compiling the model and enabling mixed-precision training at the same time
- Fix the bug where the logs cannot be properly saved to the log file after calling `torch.compile`

### New Features & Enhancements

- Add `mmpretrain` to the `MODULE2PACKAGE`. by [@mzr1996](https://github.com/mzr1996) in https://github.com/open-mmlab/mmengine/pull/1002
- Support using `get_device` in the compiled model by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/1004
- Make sure the FileHandler still alive after `torch.compile` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1021
- Unify the use of `print_log` and `logger.info(warning)` by [@LEFTeyex](https://github.com/LEFTeyex) in https://github.com/open-mmlab/mmengine/pull/997
- Publish models after training if published_keys is set in CheckpointHook by [@KerwinKai](https://github.com/KerwinKai) in https://github.com/open-mmlab/mmengine/pull/987
- Enhance the error catching in registry by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1010
- Do not print config if it is empty by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/1028

### Bug fixes

- Fix there is no space between `data_time` and metric in logs by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/1025

### Docs

- Minor fixes in EN docs to remove or replace unicode chars with ascii by [@evdcush](https://github.com/evdcush) in https://github.com/open-mmlab/mmengine/pull/1018

### Contributors

A total of 7 developers contributed to this release. Thanks [@LEFTeyex](https://github.com/LEFTeyex), [@KerwinKai](https://github.com/KerwinKai), [@mzr1996](https://github.com/mzr1996), [@evdcush](https://github.com/evdcush), [@C1rN09](https://github.com/C1rN09), [@HAOCHENYE](https://github.com/HAOCHENYE), [@zhouzaida](https://github.com/zhouzaida)

## v0.7.0 (03/16/2023)

### Highlights

- Support PyTorch 2.0! Accelerate training by compiling models. See the tutorial [Model Compilation](https://mmengine.readthedocs.io/en/latest/common_usage/speed_up_training.html#model-compilation) for details
- Add `EarlyStoppingHook` to stop training when the metric does not improve

### New Features & Enhancements

- Add configurations to support `torch.compile` in Runner by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/976
- Support `EarlyStoppingHook` by [@nijkah](https://github.com/nijkah) in https://github.com/open-mmlab/mmengine/pull/739
- Disable duplicated warning during distributed training by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/961
- Add `FUNCTIONS` root Registry by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/983
- Save the "memory" field to visualization backends by [@enkilee](https://github.com/enkilee) in https://github.com/open-mmlab/mmengine/pull/974
- Enable bf16 in `AmpOptimWrapper` by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/960
- Support writing data to `vis_backend` with prefix by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/972
- Support exporting logs of different ranks in debug mode by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/968
- Silence error when `ManagerMixin` built instance with duplicate name. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/990

### Bug fixes

- Fix optim_wrapper unittest for `pytorch < 1.10.0` by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/975
- Support calculating the flops of `matmul` with single dimension matrix by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/970
- Fix repeated warning by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/992
- Fix lint by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/993
- Fix AMP in Ascend and support using NPUJITCompile environment by [@luomaoling](https://github.com/luomaoling) in https://github.com/open-mmlab/mmengine/pull/994
- Fix inferencer gets wrong configs path by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/996

### Docs

- Translate "Debug Tricks" to English by [@enkilee](https://github.com/enkilee) in https://github.com/open-mmlab/mmengine/pull/953
- Translate "Model Analysis" document to English by [@enkilee](https://github.com/enkilee) in https://github.com/open-mmlab/mmengine/pull/956
- Translate "Model Complexity Analysis" to Chinese. by [@VoyagerXvoyagerx](https://github.com/VoyagerXvoyagerx) in https://github.com/open-mmlab/mmengine/pull/969
- Add a document about setting interval by [@YuetianW](https://github.com/YuetianW) in https://github.com/open-mmlab/mmengine/pull/964
- Translate "how to set random seed" by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/930
- Fix typo by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/965
- Fix typo in hook document by [@acdart](https://github.com/acdart) in https://github.com/open-mmlab/mmengine/pull/980
- Fix changelog date by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/986

### Contributors

A total of 10 developers contributed to this release. Thanks [@xin-li-67](https://github.com/xin-li-67), [@acdart](https://github.com/acdart), [@enkilee](https://github.com/enkilee), [@YuetianW](https://github.com/YuetianW), [@luomaoling](https://github.com/luomaoling), [@nijkah](https://github.com/nijkah), [@VoyagerXvoyagerx](https://github.com/VoyagerXvoyagerx), [@zhouzaida](https://github.com/zhouzaida), [@HAOCHENYE](https://github.com/HAOCHENYE), [@C1rN09](https://github.com/C1rN09)

## v0.6.0 (02/24/2023)

### Highlights

- Support `Apex` with `ApexOptimWrapper`
- Support analyzing model complexity.
- Add `Lion` optimizer.
- Support using environment variables in the config file.

### New Features & Enhancements

- Support model complexity computation by [@tonysy](https://github.com/tonysy) in https://github.com/open-mmlab/mmengine/pull/779
- Add Lion optimizer by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/952
- Support `Apex` with `ApexOptimWrapper` by [@xcnick](https://github.com/xcnick) in https://github.com/open-mmlab/mmengine/pull/742
- Support using environment variable in config file. by [@jbwang1997](https://github.com/jbwang1997) in https://github.com/open-mmlab/mmengine/pull/744
- Improve registry infer_scope by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/334
- Support configuring `timeout` in dist configuration by [@apacha](https://github.com/apacha) in https://github.com/open-mmlab/mmengine/pull/877
- Beautify the print result of the registry by [@Eiuyc](https://github.com/Eiuyc) in https://github.com/open-mmlab/mmengine/pull/922
- Refine the style of table by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/941
- Refine the `repr` of Registry by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/942
- Feature NPUProfilerHook by [@luomaoling](https://github.com/luomaoling) in https://github.com/open-mmlab/mmengine/pull/925
- Refactor hooks unittest by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/946
- Temporarily fix `collect_env` raise errors and stops programs by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/944
- Make sure Tensors to broadcast is contiguous by [@XWHtorrentx](https://github.com/XWHtorrentx) in https://github.com/open-mmlab/mmengine/pull/948
- Clean the UT warning caused by pytest by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/947

### Bug fixes

- Backend_args should not be modified by get_file_backend by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/897
- Support update `np.ScalarType` data in message_hub by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/898
- Support rendering Chinese character in `Visualizer` by [@KevinNuNu](https://github.com/KevinNuNu) in https://github.com/open-mmlab/mmengine/pull/887
- Fix the bug of `DefaultOptimWrapperConstructor` when the shared parameters do not require the grad by [@HIT-cwh](https://github.com/HIT-cwh) in https://github.com/open-mmlab/mmengine/pull/903

### Docs

- Add the document for the transition between IterBasedTraining and EpochBasedTraining by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/926
- Introduce how to set random seed by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/914
- Count FLOPs and parameters by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/939
- Enhance README by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/835
- Add a document about debug tricks by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/938
- Refine the format of changelog and visualization document by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/906
- Move examples to a new directory by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/911
- Resolve warnings in sphinx build by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/915
- Fix docstring by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/913
- How to set the interval parameter by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/917
- Temporarily skip errors in building pdf docs at readthedocs by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/928
- Add the links of twitter, discord, medium, and youtube by [@vansin](https://github.com/vansin) in https://github.com/open-mmlab/mmengine/pull/924
- Fix typo `shedule` by [@Dai-Wenxun](https://github.com/Dai-Wenxun) in https://github.com/open-mmlab/mmengine/pull/936
- Fix failed URL by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/943

### Contributors

A total of 15 developers contributed to this release. Thanks [@Eiuyc](https://github.com/Eiuyc), [@xcnick](https://github.com/xcnick), [@KevinNuNu](https://github.com/KevinNuNu), [@XWHtorrentx](https://github.com/XWHtorrentx), [@tonysy](https://github.com/tonysy), [@zhouzaida](https://github.com/zhouzaida), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@Dai-Wenxun](https://github.com/Dai-Wenxun), [@jbwang1997](https://github.com/jbwang1997), [@apacha](https://github.com/apacha), [@C1rN09](https://github.com/C1rN09), [@HIT-cwh](https://github.com/HIT-cwh), [@vansin](https://github.com/vansin), [@HAOCHENYE](https://github.com/HAOCHENYE), [@luomaoling](https://github.com/luomaoling).

## v0.5.0 (01/20/2023)

### Highlights

- Add `BaseInferencer` to provide a general inference interface
- Provide `ReduceOnPlateauParamScheduler` to adjust learning rate by metric
- Deprecate support for Python3.6

### New Features & Enhancements

- Deprecate support for Python3.6 by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/863
- Support non-scalar type metric value by [@mzr1996](https://github.com/mzr1996) in https://github.com/open-mmlab/mmengine/pull/827
- Remove unnecessary calls and lazily import to speed import performance by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/837
- Support `ReduceOnPlateauParamScheduler` by [@LEFTeyex](https://github.com/LEFTeyex) in https://github.com/open-mmlab/mmengine/pull/819
- Disable warning of subprocess launched by dataloader by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/870
- Add `BaseInferencer` to provide general interface by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/874

### Bug Fixes

- Fix support for Ascend device by [@wangjiangben-hw](https://github.com/wangjiangben-hw) in https://github.com/open-mmlab/mmengine/pull/847
- Fix `Config` cannot parse base config when there is `.` in tmp path, etc. `tmp/a.b/c` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/856
- Fix unloaded weights will not be initialized when using `PretrainedIinit` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/764
- Fix error package name defined in `PKG2PROJECT` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/872

### Docs

- Fix typos in `advanced_tutorials/logging.md` by [@RangeKing](https://github.com/RangeKing) in https://github.com/open-mmlab/mmengine/pull/861
- Translate CN `train_a_gan` to EN by [@yaqi0510](https://github.com/yaqi0510) in https://github.com/open-mmlab/mmengine/pull/860
- Update `fileio.md` by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/869
- Add Chinese documentation for `inferencer`. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/884

### Contributors

A total of 8 developers contributed to this release. Thanks [@LEFTeyex](https://github.com/LEFTeyex), [@RangeKing](https://github.com/RangeKing), [@yaqi0510](https://github.com/yaqi0510), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@wangjiangben-hw](https://github.com/wangjiangben-hw), [@mzr1996](https://github.com/mzr1996), [@zhouzaida](https://github.com/zhouzaida), [@HAOCHENYE](https://github.com/HAOCHENYE).

## v0.4.0 (12/28/2022)

### Highlights

- Registry supports importing modules automatically
- Upgrade the documentation and provide the **English documentation**
- Provide `ProfileHook` to profile the running process

### New Features & Enhancements

- Add `conf_path` in PetrelBackend by [@sunyc11](https://github.com/sunyc11) in https://github.com/open-mmlab/mmengine/pull/774
- Support multiple `--cfg-options`. by [@mzr1996](https://github.com/mzr1996) in https://github.com/open-mmlab/mmengine/pull/759
- Support passing arguments to `OptimWrapper.update_params` by [@twmht](https://github.com/twmht) in https://github.com/open-mmlab/mmengine/pull/796
- Make `get_torchvision_model` compatible with torch 1.13 by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/793
- Support `flat_decay_mult` and fix `bias_decay_mult` of depth-wise-conv in `DefaultOptimWrapperConstructor` by [@RangiLyu](https://github.com/RangiLyu) in https://github.com/open-mmlab/mmengine/pull/771
- Registry supports importing modules automatically. by [@RangiLyu](https://github.com/RangiLyu) in https://github.com/open-mmlab/mmengine/pull/643
- Add profiler hook functionality by [@BayMaxBHL](https://github.com/BayMaxBHL) in https://github.com/open-mmlab/mmengine/pull/768
- Make TTAModel compatible with FSDP. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/611

### Bug Fixes

- `hub.get_model` fails on some MMCls models by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/784
- Fix `BaseModel.to` and `BaseDataPreprocessor.to` to make them consistent with `torch.nn.Module` by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/783
- Fix creating a new logger at PretrainedInit by [@xiexinch](https://github.com/xiexinch) in https://github.com/open-mmlab/mmengine/pull/791
- Fix `ZeroRedundancyOptimizer` ambiguous error with param groups when PyTorch \< 1.12.0 by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/818
- Fix MessageHub set resumed key repeatedly by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/839
- Add `progress` argument to `load_from_http` by [@austinmw](https://github.com/austinmw) in https://github.com/open-mmlab/mmengine/pull/770
- Ensure metrics is not empty when saving best checkpoint by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/849

### Docs

- Add `contributing.md` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/754
- Add gif to 15 min tutorial by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/748
- Refactor documentations and translate them to English by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/786
- Fix document link by [@MambaWong](https://github.com/MambaWong) in https://github.com/open-mmlab/mmengine/pull/775
- Fix typos in EN `contributing.md` by [@RangeKing](https://github.com/RangeKing) in https://github.com/open-mmlab/mmengine/pull/792
- Translate data transform docs. by [@mzr1996](https://github.com/mzr1996) in https://github.com/open-mmlab/mmengine/pull/737
- Replace markdown table with html table by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/800
- Fix wrong example in `Visualizer.draw_polygons`  by [@lyviva](https://github.com/lyviva) in https://github.com/open-mmlab/mmengine/pull/798
- Fix docstring format and rescale the images by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/802
- Fix failed link in registry by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/811
- Fix typos  by [@shanmo](https://github.com/shanmo) in https://github.com/open-mmlab/mmengine/pull/814
- Fix wrong links and typos in docs by [@shanmo](https://github.com/shanmo) in https://github.com/open-mmlab/mmengine/pull/815
- Translate `save_gpu_memory.md` by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/803
- Translate the documentation of hook design by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/780
- Fix docstring format by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/816
- Translate `registry.md` by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/817
- Update docstring of `BaseDataElement` by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/836
- Fix typo by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/841
- Update docstring of `structures` by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/840
- Translate `optim_wrapper.md` by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/833
- Fix link error in initialize tutorial. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/843
- Fix table in `initialized.md` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/844

### Contributors

A total of 16 developers contributed to this release. Thanks [@BayMaxBHL](https://github.com/BayMaxBHL), [@RangeKing](https://github.com/RangeKing), [@Xiangxu-0103](https://github.com/Xiangxu-0103), [@xin-li-67](https://github.com/xin-li-67), [@twmht](https://github.com/twmht), [@shanmo](https://github.com/shanmo), [@sunyc11](https://github.com/sunyc11), [@lyviva](https://github.com/lyviva), [@austinmw](https://github.com/austinmw), [@xiexinch](https://github.com/xiexinch), [@mzr1996](https://github.com/mzr1996), [@RangiLyu](https://github.com/RangiLyu), [@MambaWong](https://github.com/MambaWong), [@C1rN09](https://github.com/C1rN09), [@zhouzaida](https://github.com/zhouzaida), [@HAOCHENYE](https://github.com/HAOCHENYE)

## v0.3.2 (11/24/2022)

### New Features & Enhancements

- Send git errors to subprocess.PIPE by [@austinmw](https://github.com/austinmw) in https://github.com/open-mmlab/mmengine/pull/717
- Add a common `TestRunnerTestCase` to build a Runner instance. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/631
- Align the log by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/436
- Log the called order of hooks during training process by [@songyuc](https://github.com/songyuc) in https://github.com/open-mmlab/mmengine/pull/672
- Support setting `eta_min_ratio` in `CosineAnnealingParamScheduler` by [@cir7](https://github.com/cir7) in https://github.com/open-mmlab/mmengine/pull/725
- Enhance compatibility of `revert_sync_batchnorm` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/695

### Bug Fixes

- Fix `distributed_training.py` in examples by [@PingHGao](https://github.com/PingHGao) in https://github.com/open-mmlab/mmengine/pull/700
- Format the log of `CheckpointLoader.load_checkpoint` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/685
- Fix bug of CosineAnnealingParamScheduler by [@fangyixiao18](https://github.com/fangyixiao18) in https://github.com/open-mmlab/mmengine/pull/735
- Fix `add_graph` is not called bug by [@shenmishajing](https://github.com/shenmishajing) in https://github.com/open-mmlab/mmengine/pull/632
- Fix .pre-commit-config-zh-cn.yaml pyupgrade-repo github->gitee by [@BayMaxBHL](https://github.com/BayMaxBHL) in https://github.com/open-mmlab/mmengine/pull/756

### Docs

- Add English docs of BaseDataset by [@GT9505](https://github.com/GT9505) in https://github.com/open-mmlab/mmengine/pull/713
- Fix `BaseDataset` typo about lazy initialization by [@MengzhangLI](https://github.com/MengzhangLI) in https://github.com/open-mmlab/mmengine/pull/733
- Fix typo by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/734
- Translate visualization docs by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/692

## v0.3.1 (11/09/2022)

### Highlights

- Fix error when saving best checkpoint in ddp-training

### New Features & Enhancements

- Replace `print` with `print_log` for those functions called by runner by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/686

### Bug Fixes

- Fix error when saving best checkpoint in ddp-training by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/682

### Docs

- Refine Chinese tutorials by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/694
- Add MMEval in README by [@sanbuphy](https://github.com/sanbuphy) in https://github.com/open-mmlab/mmengine/pull/669
- Fix error URL in runner docstring by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/668
- Fix error evaluator type name in `evaluator.md` by [@sanbuphy](https://github.com/sanbuphy) in https://github.com/open-mmlab/mmengine/pull/675
- Fix typo in `utils.md` [@sanbuphy](https://github.com/sanbuphy) in https://github.com/open-mmlab/mmengine/pull/702

## v0.3.0 (11/02/2022)

### New Features & Enhancements

- Support running on Ascend chip by [@wangjiangben-hw](https://github.com/wangjiangben-hw) in https://github.com/open-mmlab/mmengine/pull/572
- Support torch `ZeroRedundancyOptimizer` by [@nijkah](https://github.com/nijkah) in https://github.com/open-mmlab/mmengine/pull/551
- Add non-blocking feature to `BaseDataPreprocessor` by [@shenmishajing](https://github.com/shenmishajing) in https://github.com/open-mmlab/mmengine/pull/618
- Add documents for `clip_grad`, and support clip grad by value. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/513
- Add ROCm info when collecting env by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/633
- Add a function to mark the deprecated function. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/609
- Call `register_all_modules`  in `Registry.get()` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/541
- Deprecate `_save_to_state_dict` implemented in mmengine by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/610
- Add `ignore_keys` in ConcatDataset by [@BIGWangYuDong](https://github.com/BIGWangYuDong) in https://github.com/open-mmlab/mmengine/pull/556

### Docs

- Fix cannot show `changelog.md` in chinese documents. by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/606
- Fix Chinese docs whitespaces by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/521
- Translate installation and 15_min by [@xin-li-67](https://github.com/xin-li-67) in https://github.com/open-mmlab/mmengine/pull/629
- Refine chinese doc by [@Tau-J](https://github.com/Tau-J) in https://github.com/open-mmlab/mmengine/pull/516
- Add MMYOLO link in README by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/634
- Add MMEngine logo in docs by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/641
- Fix docstring of `BaseDataset` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/656
- Fix docstring and documentation used for `hub.get_model` by [@zengyh1900](https://github.com/zengyh1900) in https://github.com/open-mmlab/mmengine/pull/659
- Fix typo in `docs/zh_cn/advanced_tutorials/visualization.md` by [@MambaWong](https://github.com/MambaWong) in https://github.com/open-mmlab/mmengine/pull/616
- Fix typo docstring of `DefaultOptimWrapperConstructor` by [@triple-Mu](https://github.com/triple-Mu) in https://github.com/open-mmlab/mmengine/pull/644
- Fix typo in advanced tutorial by [@cxiang26](https://github.com/cxiang26) in https://github.com/open-mmlab/mmengine/pull/650
- Fix typo in `Config` docstring by [@sanbuphy](https://github.com/sanbuphy) in https://github.com/open-mmlab/mmengine/pull/654
- Fix typo in `docs/zh_cn/tutorials/config.md` by [@Xiangxu-0103](https://github.com/Xiangxu-0103) in https://github.com/open-mmlab/mmengine/pull/596
- Fix typo in `docs/zh_cn/tutorials/model.md` by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/598

### Bug Fixes

- Fix error calculation of `eta_min` in `CosineRestartParamScheduler` by [@Z-Fran](https://github.com/Z-Fran) in https://github.com/open-mmlab/mmengine/pull/639
- Fix `BaseDataPreprocessor.cast_data` could not handle string data by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/602
- Make `autocast` compatible with mps by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/587
- Fix error format of log message by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/508
- Fix error implementation of `is_model_wrapper` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/640
- Fix `VisBackend.add_config` is not called by [@shenmishajing](https://github.com/shenmishajing) in https://github.com/open-mmlab/mmengine/pull/613
- Change `strict_load` of EMAHook to False by default by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/642
- Fix `open` encoding problem of Config in Windows by [@sanbuphy](https://github.com/sanbuphy) in https://github.com/open-mmlab/mmengine/pull/648
- Fix the total number of iterations in log is a float number. by [@jbwang1997](https://github.com/jbwang1997) in https://github.com/open-mmlab/mmengine/pull/604
- Fix `pip upgrade` CI by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/622

### New Contributors

- [@shenmishajing](https://github.com/shenmishajing) made their first contribution in https://github.com/open-mmlab/mmengine/pull/618
- [@Xiangxu-0103](https://github.com/Xiangxu-0103) made their first contribution in https://github.com/open-mmlab/mmengine/pull/596
- [@Tau-J](https://github.com/Tau-J) made their first contribution in https://github.com/open-mmlab/mmengine/pull/516
- [@wangjiangben-hw](https://github.com/wangjiangben-hw) made their first contribution in https://github.com/open-mmlab/mmengine/pull/572
- [@triple-Mu](https://github.com/triple-Mu) made their first contribution in https://github.com/open-mmlab/mmengine/pull/644
- [@sanbuphy](https://github.com/sanbuphy) made their first contribution in https://github.com/open-mmlab/mmengine/pull/648
- [@Z-Fran](https://github.com/Z-Fran) made their first contribution in https://github.com/open-mmlab/mmengine/pull/639
- [@BIGWangYuDong](https://github.com/BIGWangYuDong) made their first contribution in https://github.com/open-mmlab/mmengine/pull/556
- [@zengyh1900](https://github.com/zengyh1900) made their first contribution in https://github.com/open-mmlab/mmengine/pull/659

## v0.2.0 (10/11/2022)

### New Features & Enhancements

- Add SMDDP backend and support running on AWS by [@austinmw](https://github.com/austinmw) in https://github.com/open-mmlab/mmengine/pull/579
- Refactor `FileIO` but without breaking bc by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/533
- Add test time augmentation base model by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/538
- Use `torch.lerp\_()` to speed up EMA by [@RangiLyu](https://github.com/RangiLyu) in https://github.com/open-mmlab/mmengine/pull/519
- Support converting `BN` to `SyncBN` by config by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/506
- Support defining metric name in wandb backend by [@okotaku](https://github.com/okotaku) in https://github.com/open-mmlab/mmengine/pull/509
- Add dockerfile by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/347

### Docs

- Fix API files of English documentation by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/525
- Fix typo in `instance_data.py` by [@Dai-Wenxun](https://github.com/Dai-Wenxun) in https://github.com/open-mmlab/mmengine/pull/530
- Fix the docstring of the model sub-package by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/573
- Fix a spelling error in docs/zh_cn by [@cxiang26](https://github.com/cxiang26) in https://github.com/open-mmlab/mmengine/pull/548
- Fix typo in docstring by [@MengzhangLI](https://github.com/MengzhangLI) in https://github.com/open-mmlab/mmengine/pull/527
- Update `config.md` by [@Zhengfei-0311](https://github.com/Zhengfei-0311) in https://github.com/open-mmlab/mmengine/pull/562

### Bug Fixes

- Fix `LogProcessor` does not smooth loss if the name of loss doesn't start with `loss` by [@liuyanyi](https://github.com/liuyanyi) in
  https://github.com/open-mmlab/mmengine/pull/539
- Fix failed to enable `detect_anomalous_params` in `MMSeparateDistributedDataParallel` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/588
- Fix CheckpointHook behavior unexpected if given `filename_tmpl` argument by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/518
- Fix error argument sequence in `FSDP` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/520
- Fix uploading image in wandb backend [@okotaku](https://github.com/okotaku) in https://github.com/open-mmlab/mmengine/pull/510
- Fix loading state dictionary in `EMAHook` by [@okotaku](https://github.com/okotaku) in https://github.com/open-mmlab/mmengine/pull/507
- Fix circle import in `EMAHook` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/523
- Fix unit test could fail caused by `MultiProcessTestCase`  by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/535
- Remove unnecessary "if statement" in `Registry` by [@MambaWong](https://github.com/MambaWong) in https://github.com/open-mmlab/mmengine/pull/536
- Fix `_save_to_state_dict` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/542
- Support comparing NumPy array dataset meta in `Runner.resume` by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/511
- Use `get` instead of `pop` to dump `runner_type` in `build_runner_from_cfg` by [@nijkah](https://github.com/nijkah) in https://github.com/open-mmlab/mmengine/pull/549
- Upgrade pre-commit hooks by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/576
- Delete the error comment in `registry.md` by [@vansin](https://github.com/vansin) in https://github.com/open-mmlab/mmengine/pull/514
- Fix Some out-of-date unit tests by [@C1rN09](https://github.com/C1rN09) in https://github.com/open-mmlab/mmengine/pull/586
- Fix typo in `MMFullyShardedDataParallel` by [@yhna940](https://github.com/yhna940) in https://github.com/open-mmlab/mmengine/pull/569
- Update Github Action CI and CircleCI by [@zhouzaida](https://github.com/zhouzaida) in https://github.com/open-mmlab/mmengine/pull/512
- Fix unit test in windows by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/515
- Fix merge ci & multiprocessing unit test by [@HAOCHENYE](https://github.com/HAOCHENYE) in https://github.com/open-mmlab/mmengine/pull/529

### New Contributors

- [@okotaku](https://github.com/okotaku) made their first contribution in https://github.com/open-mmlab/mmengine/pull/510
- [@MengzhangLI](https://github.com/MengzhangLI) made their first contribution in https://github.com/open-mmlab/mmengine/pull/527
- [@MambaWong](https://github.com/MambaWong) made their first contribution in https://github.com/open-mmlab/mmengine/pull/536
- [@cxiang26](https://github.com/cxiang26) made their first contribution in https://github.com/open-mmlab/mmengine/pull/548
- [@nijkah](https://github.com/nijkah) made their first contribution in https://github.com/open-mmlab/mmengine/pull/549
- [@Zhengfei-0311](https://github.com/Zhengfei-0311) made their first contribution in https://github.com/open-mmlab/mmengine/pull/562
- [@austinmw](https://github.com/austinmw) made their first contribution in https://github.com/open-mmlab/mmengine/pull/579
- [@yhna940](https://github.com/yhna940) made their first contribution in https://github.com/open-mmlab/mmengine/pull/569
- [@liuyanyi](https://github.com/liuyanyi) made their first contribution in https://github.com/open-mmlab/mmengine/pull/539
