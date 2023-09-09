# Train a Segmentation Model

## Download Camvid Dataset

First, you should get the collated Camvid dataset on OpenDataLab to use for the segmentation training example. The official download steps are shown below.

```bash
# https://opendatalab.com/CamVid
# Configure install
pip install opendatalab
# Upgraded version
pip install -U opendatalab
# Login
odl login
# Download this dataset
mkdir data
odl get CamVid -d data
# Preprocess data in Linux. You should extract the files to data manually in
# Windows
tar -xzvf data/CamVid/raw/CamVid.tar.gz.00 -C ./data
```

## Run the Example

Single device training

```bash
python examples/segmentation/train.py
```

Distributed data parallel training

```bash
tochrun -nnodes 1 -nproc_per_node 8 examples/segmentation/train.py --launcher pytorch
```
