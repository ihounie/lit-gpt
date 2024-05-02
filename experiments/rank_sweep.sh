for r in 6 12 24 48 96 192 384 768
do
    CUDA_VISIBLE_DEVICES=0 python finetune/lora.py --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" --precision "bf16-true" --rank $r --tensor_lora 'True' --joint_layers 'True' --joint_qkvp 'True' --joint_heads 'True' --init_scale 0.0 --alpha 16 --lora_dropout 0.05 --learning_rate 0.0004
done