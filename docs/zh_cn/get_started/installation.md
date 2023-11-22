# 安装

## 环境依赖

- Python 3.7+
- PyTorch 1.6+
- CUDA 9.2+
- GCC 5.4+

## 准备环境

1. 使用 conda 新建虚拟环境，并进入该虚拟环境；

   ```bash
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. 安装 PyTorch

   在安装 MMEngine 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/#start-locally)。使用以下命令验证 PyTorch 是否安装

   ```bash
   python -c 'import torch;print(torch.__version__)'
   ```

## 安装 MMEngine

:::{note}
如果你只想使用 MMEngine 中的 fileio、registry 和 config 模块，你可以安装`mmengine-lite`，它只会安装必须的几个第三方库依赖（例如不会安装 opencv、matplotlib）：

```bash
pip install mmengine-lite
```

:::

### 使用 mim 安装

[mim](https://github.com/open-mmlab/mim) 是 OpenMMLab 项目的包管理工具，使用它可以很方便地安装 OpenMMLab 项目。

```bash
pip install -U openmim
mim install mmengine
```

### 使用 pip 安装

```bash
pip install mmengine
```

### 使用 docker 镜像

1. 构建镜像

   ```bash
   docker build -t mmengine https://github.com/open-mmlab/mmengine.git#main:docker/release
   ```

   更多构建方式请参考 [mmengine/docker](https://github.com/open-mmlab/mmengine/tree/main/docker)。

2. 运行镜像

   ```bash
   docker run --gpus all --shm-size=8g -it mmengine
   ```

### 源码安装

#### 安装 mmengine

```bash
# 如果克隆代码仓库的速度过慢，可以从 https://gitee.com/open-mmlab/mmengine.git 克隆
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```

#### 安装 mmengine-lite

```bash
# 如果克隆代码仓库的速度过慢，可以从 https://gitee.com/open-mmlab/mmengine.git 克隆
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
MMENGINE_LITE=1 pip install -e . -v
```

## 验证安装

为了验证是否正确安装了 MMEngine 和所需的环境，我们可以运行以下命令

```bash
python -c 'import mmengine;print(mmengine.__version__)'
```
