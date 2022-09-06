# Docker images

There are two `Dockerfile` files to build docker images, one to build an image with the mmengine package and the other with the mmengine development environment.

```text
.
|-- README.md
|-- dev  # build with mmengine development environment
|   `-- Dockerfile
`-- release  # build with mmengine package
    `-- Dockerfile
```

## Build docker images

### Build with mmengine package

Build with local repository

```bash
git clone https://github.com/open-mmlab/mmengine.git && cd mmengine
docker build -t mmengine -f docker/release/Dockerfile .
```

Or build with remote repository

```bash
docker build -t mmengine https://github.com/open-mmlab/mmengine.git#main:docker/release
```

The [Dockerfile](release/Dockerfile) installs the latest released version of mmengine by default, but you can specify mmengine versions to install expected versions.

```bash
docker image build -t mmengine -f docker/release/Dockerfile --build-arg MMENGINE=0.1.0 .
```

If you also want to use other versions of PyTorch and CUDA, you can also pass them when building docker images.

An example to build an image with PyTorch 1.11 and CUDA 11.3.

```bash
docker build -t mmengine -f docker/release/Dockerfile \
    --build-arg PYTORCH=1.9.0 \
    --build-arg CUDA=11.1 \
    --build-arg CUDNN=8 .
```

More available versions of PyTorch and CUDA can be found at [dockerhub/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags).

### Build with mmengine development environment

If you want to build an docker image with the mmengine development environment, you can use the following command

```bash
git clone https://github.com/open-mmlab/mmengine.git && cd mmengine
docker build -t mmengine -f docker/dev/Dockerfile .
```

## Run images

```bash
docker run --gpus all --shm-size=8g -it mmengine
```

See [docker run](https://docs.docker.com/engine/reference/commandline/run/) for more usages.
