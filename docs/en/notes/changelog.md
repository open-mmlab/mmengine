# Changelog of v0.x

## v0.2.0 (11/10/2022)

### New Features & Enhancements

- Support defining metric name in wandb backend @okotaku in https://github.com/open-mmlab/mmengine/pull/509
- Use `torch.lerp\_()` to speed up EMA by @RangiLyu in https://github.com/open-mmlab/mmengine/pull/519
- Add test time augmentation base model by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/538
- Support converting `BN` to `SyncBN` by config by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/506
- Refactor `FileIO` but without breaking bc by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/533
- Add dockerfile by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/347

### Docs

- Fix API files of en docs by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/525
- Fix typo in `instance_data.py` by @Dai-Wenxun in https://github.com/open-mmlab/mmengine/pull/530
- Fix the docstring of the model sub-package by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/573

### Bug Fixes

- Update Github Action CI and CircleCI by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/512
- Fix upload image in wandb backend @okotaku in https://github.com/open-mmlab/mmengine/pull/510
- Fix loading state dictionary in EMAHook @okotaku in https://github.com/open-mmlab/mmengine/pull/507
- Fix error argument sequence in `FSDP` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/520
- Fix circle import in EMAHook by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/523
- Fix unit test in windows by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/515
- Fix typo in docstring by @MengzhangLI in https://github.com/open-mmlab/mmengine/pull/527
- Fix merge ci & multiprocessing unit test by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/529
- Fix unit test could fail caused by `MultiProcessTestCase`  by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/535
- Remove unnecessary "if statement" by @MambaWong in https://github.com/open-mmlab/mmengine/pull/536
- Fix CheckpointHook behavior unexpected if given `filename_tmpl` argument by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/518
- Fix `_save_to_state_dict` by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/542
- Support compare NumPy array dataset meta by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/511
- Fix a spelling error in docs/zh_cn by @cxiang26 in https://github.com/open-mmlab/mmengine/pull/548
- Use `get` instead of `pop` to dump `runner_type` by @nijkah in https://github.com/open-mmlab/mmengine/pull/549
- Upgrade pre-commit hooks by @zhouzaida in https://github.com/open-mmlab/mmengine/pull/576
- Update `config.md` by @Zhengfei-0311 in https://github.com/open-mmlab/mmengine/pull/562
- Delete the error comment by @vansin in https://github.com/open-mmlab/mmengine/pull/514
- Fix detect_anomalous_params by @HAOCHENYE in https://github.com/open-mmlab/mmengine/pull/588
- Some out-of-date unit tests by @C1rN09 in https://github.com/open-mmlab/mmengine/pull/586
- Add smddp dist backend option by @austinmw in https://github.com/open-mmlab/mmengine/pull/579
- Fix typo by @yhna940 in https://github.com/open-mmlab/mmengine/pull/569
- Fix loss smooth when the loss name doesn't start with `loss` by @liuyanyi in
  https://github.com/open-mmlab/mmengine/pull/539

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
