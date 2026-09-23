# MintFlow Image Editing

This repository contains the code for image editing with **MintFlow**.

## Installation

Install the required dependencies with:

    pip install -r requirements.txt

## Pretrained Models and Baselines

The general pipeline and baseline implementations used in our experiments are directly adopted from the official [FlowChef](https://github.com/FlowChef/flowchef) repository without modification.

## Running Image Editing

Run the following script to reproduce the image editing experiments:

    bash edit.sh

To use a different input image, place the image and its corresponding mask in `/data/images` and `/data/masks`, respectively. Then, modify the `data` variable in `edit.sh` accordingly.

You should also update the `prompt` corresponding to the input image and specify the desired editing prompt (`edit_prompt`) in `edit.sh`.

Then run:

    bash edit.sh

## Acknowledgements

We thank the authors of [FlowChef](https://github.com/FlowChef/flowchef) for publicly releasing the implementations used in this work.
