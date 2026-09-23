#!/bin/bash

ckpt='./checkpoints/afhq-configF.pth'
cfg='./configs_unet/afhq64_ve_aug.json'
data='./data/AFHQ-Cat'

gpu=0

for problem in "box_inpaint" "super_resolution" "deblur"; do
    case $problem in
        "box_inpaint")
            params="--mask_size 20"
            ;;
        "super_resolution")
            params="--scale_factor 4"
            ;;
        "deblur")
            params="--kernel_size 11 --blur_sigma 1.0"
            ;;
    esac

    python generate_inverseproblems_dflow.py \
        --gpu 0 \
        --solver euler \
        --N 10 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/dflow_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        --mask_size 20 \
        --gradient_scale 500

    python generate_inverseproblems_dps.py \
        --gpu 0 \
        --solver euler \
        --N 100 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/dps_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        ${params} \
        --gradient_scale 50


    python generate_inverseproblems_freedom.py \
        --gpu 0 \
        --solver euler \
        --N 50 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/freedom_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        ${params} \
        --gradient_scale 50


    python generate_inverseproblems_pnpflow.py \
        --gpu 0 \
        --solver euler \
        --N 50 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/pnpflow_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        ${params} \
        --gradient_scale 500


    python generate_inverseproblems_flowchef.py \
        --gpu 0 \
        --solver euler \
        --N 10 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/flowchef_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        ${params} \
        --gradient_scale 500


    python generate_inverseproblems_mintflow.py \
        --gpu 0 \
        --solver euler \
        --N 10 \
        --sampler new \
        --batchsize 1 \
        --ckpt "${ckpt}" \
        --config "${cfg}" \
        --input_dir "${data}" \
        --dir "./outputs/inverseproblems/adjoint_AFHQCat_${problem}" \
        --inverse_problem "${problem}" \
        --noise_sigma 0.0 \
        ${params}


done