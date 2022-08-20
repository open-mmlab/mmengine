# Installation

## Prerequisites

- Python 3.6+
- PyTorch 1.6+
- CUDA 9.2+
- GCC 5.4+

## Prepare Environment

1. Create a conda environment and activate it

   ```bash
   conda create -n open-mmlab python=3.7 -y
   conda activate open-mmlab
   ```

2. Install PyTorch

   Before installing MMEngine, make sure that PyTorch has been successfully installed following the [official guide](https://pytorch.org/). Using the following command to verify whether PyTorch is installed correctly

   ```bash
   python -c 'import torch;print(torch.__version__)'
   ```

## Install MMEngine

### Install with mim

[mim](https://github.com/open-mmlab/mim) is a package management tool for the OpenMMLab project, which can be used to easily install the OpenMMLab project.

```bash
pip install -U openmim
mim install mmengine
```

### Install with pip

```bash
pip install mmengine
```

### Using MMEngine with Docker

1. Build docker image

   ```bash
   docker build -t mmengine https://github.com/open-mmlab/mmengine.git#main:docker/release
   ```

   For more build methods, please refer to [mmengine/docker](https://github.com/open-mmlab/mmengine/tree/main/docker).

2. Run docker image

   ```bash
   docker run --gpus all --shm-size=8g -it mmengine
   ```

### Build from source

```bash
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e . -v
```

## Verify the installation

To verify whether MMEngine is installed correctly, we can run the following command:

```bash
python -c 'import mmengine;print(mmengine.__version__)'
```
