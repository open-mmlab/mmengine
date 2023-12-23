# Installation

## Prerequisites

- Python 3.7+
- PyTorch 1.6+
- CUDA 9.2+
- GCC 5.4+

## Prepare the Environment

1. Use conda and activate the environment:

   ```bash
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. Install PyTorch

   Before installing `MMEngine`, please make sure that PyTorch has been successfully installed in the environment. You can refer to [PyTorch official installation documentation](https://pytorch.org/get-started/locally/#start-locally). Verify the installation with the following command:

   ```bash
   python -c 'import torch;print(torch.__version__)'
   ```

## Install MMEngine

:::{note}
If you only want to use the fileio, registry, and config modules in MMEngine, you can install `mmengine-lite`, which will only install the few third-party library dependencies that are necessary (e.g., it will not install opencv, matplotlib):

```bash
pip install mmengine-lite
```

:::

### Install with mim

[mim](https://github.com/open-mmlab/mim) is a package management tool for OpenMMLab projects, which can be used to install the OpenMMLab project easily.

```bash
pip install -U openmim
mim install mmengine
```

### Install with pip

```bash
pip install mmengine
```

### Use docker images

1. Build the image

   ```bash
   docker build -t mmengine https://github.com/open-mmlab/mmengine.git#main:docker/release
   ```

   More information can be referred from [mmengine/docker](https://github.com/open-mmlab/mmengine/tree/main/docker).

2. Run the image

   ```bash
   docker run --gpus all --shm-size=8g -it mmengine
   ```

### Build from source

#### Build mmengine

```bash
# if cloning speed is too slow, you can switch the source to https://gitee.com/open-mmlab/mmengine.git
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```

#### Build mmengine-lite

```bash
# if cloning speed is too slow, you can switch the source to https://gitee.com/open-mmlab/mmengine.git
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
MMENGINE_LITE=1 pip install -e . -v
```

## Verify the Installation

To verify if `MMEngine` and the necessary environment are successfully installed, we can run this command:

```bash
python -c 'import mmengine;print(mmengine.__version__)'
```
