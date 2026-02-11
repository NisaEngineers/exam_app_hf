#!/usr/bin/env python3
"""
download_model.py — Pre-download the HuggingFace model to the local ./models/ folder.

Run this ONCE before starting the server:
    python download_model.py

After this, the server works fully offline.
"""

import os
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_ID  = "google/gemma-3-4b-it"
MODEL_DIR = Path("./models") / MODEL_ID.replace("/", "--")

def check_disk_space(required_gb: float = 12.0):
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / 1e9
    log.info(f"Free disk space: {free_gb:.1f} GB  (required ≈ {required_gb} GB)")
    if free_gb < required_gb:
        log.error(
            f"Not enough disk space! Need ~{required_gb} GB, have {free_gb:.1f} GB."
        )
        sys.exit(1)

def download():
    log.info("=" * 60)
    log.info(f"Model  : {MODEL_ID}")
    log.info(f"Cache  : {MODEL_DIR.resolve()}")
    log.info("=" * 60)

    check_disk_space(required_gb=12.0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # HF token (needed for gated models like Gemma)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        log.warning(
            "⚠️  HF_TOKEN not set. Gemma is a gated model — you may need to:\n"
            "  1. Accept the licence at https://huggingface.co/google/gemma-3-4b-it\n"
            "  2. Export HF_TOKEN=<your_token> before running this script.\n"
            "  (Proceeding anyway — will fail if not authorised.)"
        )

    log.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_DIR),
        token=hf_token,
        trust_remote_code=True,
    )
    log.info("✅ Tokenizer downloaded.")

    log.info("Downloading model weights (this may take a while)...")
    # Download in bfloat16 to save disk space (~8 GB vs 16 GB for float32)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_DIR),
        token=hf_token,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log.info(f"✅ Model downloaded to: {MODEL_DIR.resolve()}")

    # Quick sanity test
    log.info("Running quick sanity test...")
    inputs = tokenizer("Hello", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    log.info(f"Sanity check output: {decoded!r}")
    log.info("✅ Model is working correctly.")
    log.info("")
    log.info("You can now start the server with:  python main.py  or  uvicorn main:app --reload")

if __name__ == "__main__":
    download()
