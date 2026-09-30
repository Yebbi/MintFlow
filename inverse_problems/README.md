# MintFlow Image Inverse Problems

This repository contains the code for image inverse problems with **MintFlow**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/643f3c91-ad71-49b9-8a0d-923a17520d22" width="90%">
</p>

We consider three image inverse problems:

* Inpainting

* Super-resolution

* Deblurring


## Installation

Install the required dependencies with:

    pip install -r requirements.txt

## Pretrained Models

Download the pretrained Rectified Flow model:

    mkdir -p ./checkpoints/
    wget -c https://huggingface.co/wangfuyun/Rectified-Diffusion/resolve/main/weights/rd.ckpt -P ./checkpoints/

## Baselines

The baseline implementations used in our experiments are directly adopted from the official [FlowChef](https://github.com/FlowChef/flowchef) repository without modification.


## Running Image Inverse Problems

Run the following script to reproduce the image inverse-problem experiments:

    bash inverseproblems.sh

By default, the script uses the following data directory:

    data='./data/AFHQ-Cat'

To use a different dataset, modify the `data` variable in `inverseproblems.sh`. For example:

    data='./data/YOUR_DATASET'

Then run:

    bash inverseproblems.sh

## Acknowledgements

We thank the authors of [FlowChef](https://github.com/FlowChef/flowchef) for publicly releasing the baseline implementations used in this work.
