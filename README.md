# ⚡ LoRTA: Tensor Parametrizations


## Setup

Clone the repo

```bash
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```

Install dependencies

```bash
pip install -r requirements.txt tokenizers sentencepiece
```

All of our bash scripts are in the folder experiments.

Download the alpaca dataset and LLAMA 7B model:
```bash
./experiments/prepare_llama7b_alpaca.sh
```

The baseline lora can be run using
```bash
./experiments/lora-bline.sh
```

A tensor lora example experiment can be found at
```bash
./experiments/lora-bline.sh
```


## What I want to fine-tune/play around with

* `alpha`: this hyperparameter scales the update ($W' = W+\alpha/r * dW$). Default is 16, but it gets scaled inversely by the matrix rank in OG LORA.
* `lora_dropout`: adapter dropout - we might need less or even no regularization. Default is 0.05.
* `learning rate`: Default is 3e-4. AdamW shouldn't be overly sensitive to this. But training curves are suuper noisy for both OG lora and our method.