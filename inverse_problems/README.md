# MintFlow Image Inverse Problems

This repository contains the code for image inverse problems with **MintFlow**.

## Installation

Install the required dependencies with:

    pip install -r requirements.txt

## Pretrained Models and Baselines

The pretrained Rectified Flow models and baseline implementations used in our experiments are directly adopted from the official [FlowChef](https://github.com/FlowChef/flowchef) repository without modification.

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

We thank the authors of [FlowChef](https://github.com/FlowChef/flowchef) for publicly releasing the pretrained Rectified Flow models and baseline implementations used in this work.
