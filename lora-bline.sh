for r in 1 20 200 2000
do
    CUDA_VISIBLE_DEVICES=1 python finetune/lora.py --precision "bf16-true" --rank $r --joint_qkvp 'False'
done