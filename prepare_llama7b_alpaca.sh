python scripts/download.py --repo_id openlm-research/open_llama_7b
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_7b
python scripts/prepare_alpaca.py --checkpoint_dir checkpoints/openlm-research/open_llama_7b