for r in 1
do
    python finetune/lora.py --precision "bf16-true" --rank $r
done