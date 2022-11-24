# Changelog of v0.x

## v0.3.2 (11/24/2022)

### New Features & Enhancements

- Send git errors to subprocess.PIPE by @austinmw in https://github.com/open-mmlab/mmengine/pull/717
- Add a common `TestRunnerTestCase` to build a Runner instance. by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/631
- Align the log by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/436
- Log the called order of hooks during training process by @songyuc in https://github.com/open-mmlab/mmengine/pull/672
- Support setting `eta_min_ratio` in `CosineAnnealingParamScheduler` by @cir7 in https://github.com/open-mmlab/mmengine/pull/725
- Enhance compatibility of `revert_sync_batchnorm` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/695

### Bug Fixes

- Fix `distributed_training.py` in examples by @PingHGao in https://github.com/open-mmlab/mmengine/pull/700
- Format the log of `CheckpointLoader.load_checkpoint` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/685
- Fix bug of CosineAnnealingParamScheduler by @fangyixiao18 in https://github.com/open-mmlab/mmengine/pull/735
- Fix `add_graph` is not called bug by @shenmishajing in https://github.com/open-mmlab/mmengine/pull/632
- Fix .pre-commit-config-zh-cn.yaml pyupgrade-repo github->gitee by @BayMaxBHL in https://github.com/open-mmlab/mmengine/pull/756

### Docs

- Add English docs of BaseDataset by @GT9505 in https://github.com/open-mmlab/mmengine/pull/713
- Fix `BaseDataset` typo about lazy initialization by @MengzhangLI in https://github.com/open-mmlab/mmengine/pull/733
- Fix typo by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/734
- Translate visualization docs by @xin-li-67 in https://github.com/open-mmlab/mmengine/pull/692

## v0.3.1 (11/09/2022)

### Highlights

- Fix error when saving best checkpoint in ddp-training

### New Features & Enhancements

- Replace `print` with `print_log` for those functions called by runner by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/686

### Bug Fixes

- Fix error when saving best checkpoint in ddp-training by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/682

### Docs

- Refine Chinese tutorials by @Xiangxu-0103 in https://github.com/open-mmlab/mmengine/pull/694
- Add MMEval in README by @sanbuphy in https://github.com/open-mmlab/mmengine/pull/669
- Fix error URL in runner docstring by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/668
- Fix error evaluator type name in `evaluator.md` by @sanbuphy in https://github.com/open-mmlab/mmengine/pull/675
- Fix typo in `utils.md` @sanbuphy in https://github.com/open-mmlab/mmengine/pull/702

## v0.3.0 (11/02/2022)

### New Features & Enhancements

- Support running on Ascend chip by @wangjiangben-hw in https://github.com/open-mmlab/mmengine/pull/572
- Support torch `ZeroRedundancyOptimizer` by @nijkah in https://github.com/open-mmlab/mmengine/pull/551
- Add non-blocking feature to `BaseDataPreprocessor` by @shenmishajing in https://github.com/open-mmlab/mmengine/pull/618
- Add documents for `clip_grad`, and support clip grad by value. by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/513
- Add ROCm info when collecting env by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/633
- Add a function to mark the deprecated function. by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/609
- Call `register_all_modules`  in `Registry.get()` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/541
- Deprecate `_save_to_state_dict` implemented in mmengine by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/610
- Add `ignore_keys` in ConcatDataset by @BIGWangYuDong in https://github.com/open-mmlab/mmengine/pull/556

### Docs

- Fix cannot show `changelog.md` in chinese documents. by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/606
- Fix Chinese docs whitespaces by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/521
- Translate installation and 15_min by @xin-li-67 in https://github.com/open-mmlab/mmengine/pull/629
- Refine chinese doc by @Tau-J in https://github.com/open-mmlab/mmengine/pull/516
- Add MMYOLO link in README by @Xiangxu-0103 in https://github.com/open-mmlab/mmengine/pull/634
- Add MMEngine logo in docs by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/641
- Fix docstring of `BaseDataset` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/656
- Fix docstring and documentation used for `hub.get_model` by @zengyh1900 in https://github.com/open-mmlab/mmengine/pull/659
- Fix typo in `docs/zh_cn/advanced_tutorials/visualization.md` by @MambaWong in https://github.com/open-mmlab/mmengine/pull/616
- Fix typo docstring of `DefaultOptimWrapperConstructor` by @triple-Mu in https://github.com/open-mmlab/mmengine/pull/644
- Fix typo in advanced tutorial by @cxiang26 in https://github.com/open-mmlab/mmengine/pull/650
- Fix typo in `Config` docstring by @sanbuphy in https://github.com/open-mmlab/mmengine/pull/654
- Fix typo in `docs/zh_cn/tutorials/config.md` by @Xiangxu-0103 in https://github.com/open-mmlab/mmengine/pull/596
- Fix typo in `docs/zh_cn/tutorials/model.md` by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/598

### Bug Fixes

- Fix error calculation of `eta_min` in `CosineRestartParamScheduler` by @Z-Fran in https://github.com/open-mmlab/mmengine/pull/639
- Fix `BaseDataPreprocessor.cast_data` could not handle string data by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/602
- Make `autocast` compatible with mps by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/587
- Fix error format of log message by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/508
- Fix error implementation of `is_model_wrapper` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/640
- Fix `VisBackend.add_config` is not called by @shenmishajing in https://github.com/open-mmlab/mmengine/pull/613
- Change `strict_load` of EMAHook to False by default by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/642
- Fix `open` encoding problem of Config in Windows by @sanbuphy in https://github.com/open-mmlab/mmengine/pull/648
- Fix the total number of iterations in log is a float number. by @jbwang1997 in https://github.com/open-mmlab/mmengine/pull/604
- Fix `pip upgrade` CI by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/622

### New Contributors

- @shenmishajing made their first contribution in https://github.com/open-mmlab/mmengine/pull/618
- @Xiangxu-0103 made their first contribution in https://github.com/open-mmlab/mmengine/pull/596
- @Tau-J made their first contribution in https://github.com/open-mmlab/mmengine/pull/516
- @wangjiangben-hw made their first contribution in https://github.com/open-mmlab/mmengine/pull/572
- @triple-Mu made their first contribution in https://github.com/open-mmlab/mmengine/pull/644
- @sanbuphy made their first contribution in https://github.com/open-mmlab/mmengine/pull/648
- @Z-Fran made their first contribution in https://github.com/open-mmlab/mmengine/pull/639
- @BIGWangYuDong made their first contribution in https://github.com/open-mmlab/mmengine/pull/556
- @zengyh1900 made their first contribution in https://github.com/open-mmlab/mmengine/pull/659

## v0.2.0 (11/10/2022)

### New Features & Enhancements

- Add SMDDP backend and support running on AWS by @austinmw in https://github.com/open-mmlab/mmengine/pull/579
- Refactor `FileIO` but without breaking bc by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/533
- Add test time augmentation base model by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/538
- Use `torch.lerp\_()` to speed up EMA by @RangiLyu in https://github.com/open-mmlab/mmengine/pull/519
- Support converting `BN` to `SyncBN` by config by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/506
- Support defining metric name in wandb backend by @okotaku in https://github.com/open-mmlab/mmengine/pull/509
- Add dockerfile by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/347

### Docs

- Fix API files of English documentation by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/525
- Fix typo in `instance_data.py` by @Dai-Wenxun in https://github.com/open-mmlab/mmengine/pull/530
- Fix the docstring of the model sub-package by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/573
- Fix a spelling error in docs/zh_cn by @cxiang26 in https://github.com/open-mmlab/mmengine/pull/548
- Fix typo in docstring by @MengzhangLI in https://github.com/open-mmlab/mmengine/pull/527
- Update `config.md` by @Zhengfei-0311 in https://github.com/open-mmlab/mmengine/pull/562

### Bug Fixes

- Fix `LogProcessor` does not smooth loss if the name of loss doesn't start with `loss` by @liuyanyi in
  https://github.com/open-mmlab/mmengine/pull/539
- Fix failed to enable `detect_anomalous_params` in `MMSeparateDistributedDataParallel` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/588
- Fix CheckpointHook behavior unexpected if given `filename_tmpl` argument by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/518
- Fix error argument sequence in `FSDP` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/520
- Fix uploading image in wandb backend @okotaku in https://github.com/open-mmlab/mmengine/pull/510
- Fix loading state dictionary in `EMAHook` by @okotaku in https://github.com/open-mmlab/mmengine/pull/507
- Fix circle import in `EMAHook` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/523
- Fix unit test could fail caused by `MultiProcessTestCase`  by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/535
- Remove unnecessary "if statement" in `Registry` by @MambaWong in https://github.com/open-mmlab/mmengine/pull/536
- Fix `_save_to_state_dict` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/542
- Support comparing NumPy array dataset meta in `Runner.resume` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/511
- Use `get` instead of `pop` to dump `runner_type` in `build_runner_from_cfg` by @nijkah in https://github.com/open-mmlab/mmengine/pull/549
- Upgrade pre-commit hooks by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/576
- Delete the error comment in `registry.md` by @vansin in https://github.com/open-mmlab/mmengine/pull/514
- Fix Some out-of-date unit tests by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/586
- Fix typo in `MMFullyShardedDataParallel` by @yhna940 in https://github.com/open-mmlab/mmengine/pull/569
- Update Github Action CI and CircleCI by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/512
- Fix unit test in windows by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/515
- Fix merge ci & multiprocessing unit test by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/529

### New Contributors

- @okotaku made their first contribution in https://github.com/open-mmlab/mmengine/pull/510
- @MengzhangLI made their first contribution in https://github.com/open-mmlab/mmengine/pull/527
- @MambaWong made their first contribution in https://github.com/open-mmlab/mmengine/pull/536
- @cxiang26 made their first contribution in https://github.com/open-mmlab/mmengine/pull/548
- @nijkah made their first contribution in https://github.com/open-mmlab/mmengine/pull/549
- @Zhengfei-0311 made their first contribution in https://github.com/open-mmlab/mmengine/pull/562
- @austinmw made their first contribution in https://github.com/open-mmlab/mmengine/pull/579
- @yhna940 made their first contribution in https://github.com/open-mmlab/mmengine/pull/569
- @liuyanyi made their first contribution in https://github.com/open-mmlab/mmengine/pull/539
