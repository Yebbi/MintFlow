data="000000000001" 
seed=189
steps=50
max_step=20
prompt="a cake" 
edit_prompt="a donut"

echo "========================================"
echo "edit_prompt: $edit_prompt"
echo "========================================"


python ./edit_pnpflow.py \
    --input_image "./data/images/${data}.jpg" \
    --mask_image "./data/masks/${data}.png" \
    --prompt "$prompt" \
    --edit_prompt "$edit_prompt" \
    --num_inference_steps $steps \
    --max_steps $max_step \
    --learning_rate 0.5 \
    --max_source_steps 20 \
    --optimization_steps 5 \
    --output_path "outputs/${data}/pnpflow_edit" \
    --true_cfg 2.0 \
    --seed $seed
    

python ./edit_freedom.py \
    --input_image "./data/images/${data}.jpg" \
    --mask_image "./data/masks/${data}.png" \
    --prompt "$prompt" \
    --edit_prompt "$edit_prompt" \
    --num_inference_steps $steps \
    --max_steps $max_step \
    --learning_rate 0.5 \
    --max_source_steps 20 \
    --optimization_steps 5 \
    --output_path "outputs/${data}/freedom_edit" \
    --true_cfg 2.0 \
    --seed $seed

python ./edit_dps.py \
    --input_image "./data/images/${data}.jpg" \
    --mask_image "./data/masks/${data}.png" \
    --prompt "$prompt" \
    --edit_prompt "$edit_prompt" \
    --num_inference_steps $steps \
    --max_steps $max_step \
    --learning_rate 0.5 \
    --max_source_steps 20 \
    --optimization_steps 5 \
    --gradient_scale 50.0 \
    --output_path "outputs/${data}/dps_edit" \
    --true_cfg 2.0 \
    --seed $seed

python ./edit_flowchef.py \
    --input_image "./data/images/${data}.jpg" \
    --mask_image "./data/masks/${data}.png" \
    --prompt "$prompt" \
    --edit_prompt "$edit_prompt" \
    --num_inference_steps $steps \
    --max_steps $max_step \
    --learning_rate 0.5 \
    --max_source_steps 20 \
    --optimization_steps 5 \
    --output_path "outputs/${data}/flowchef_edit" \
    --true_cfg 2.0 \
    --seed $seed

python ./edit_mintflow.py \
    --input_image "./data/images/${data}.jpg" \
    --mask_image "./data/masks/${data}.png" \
    --prompt "$prompt" \
    --edit_prompt "$edit_prompt" \
    --num_inference_steps $steps \
    --max_steps $max_step \
    --max_source_steps 20 \
    --output_path "outputs/${data}/mintflow_edit" \
    --true_cfg 2.0 \
    --seed $seed