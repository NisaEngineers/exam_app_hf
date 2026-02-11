"""
model.py — Local HuggingFace model loader and question generator.

Model: google/gemma-3-4b-it
  - Excellent Bangla + English multilingual support
  - 4B parameters fits well in 40 GB RAM (CPU) or with 4 GB VRAM (GPU offload)
  - Auto-downloaded to ./models/ on first run
  - Subsequent runs load from local cache (no internet needed)

Hardware strategy:
  - If CUDA GPU with >=4 GB VRAM is available → use device_map="auto" (splits across GPU+CPU)
  - Otherwise → pure CPU with optimised settings (bfloat16 / float32)
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_ID = "google/gemma-3-4b-it"          # Best Bangla+English 4B model
MODEL_DIR = Path("./models") / MODEL_ID.replace("/", "--")  # local cache folder

# Generation hyperparameters
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.3   # low = more factual / consistent
TOP_P          = 0.9
REPETITION_PENALTY = 1.15

# ─────────────────────────────────────────────
# Singleton – loaded once at startup
# ─────────────────────────────────────────────
_pipe = None

def _detect_device() -> str:
    """Return 'cuda' if a GPU with >=3 GB free VRAM exists, else 'cpu'."""
    if torch.cuda.is_available():
        free_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU detected: {torch.cuda.get_device_name(0)}, VRAM: {free_vram:.1f} GB")
        if free_vram >= 3.0:
            return "cuda"
        log.warning("GPU VRAM < 3 GB – falling back to CPU")
    return "cpu"

def get_model():
    """
    Load (or return cached) the text-generation pipeline.
    Downloads model to MODEL_DIR on first call.
    """
    global _pipe
    if _pipe is not None:
        return _pipe

    device = _detect_device()
    log.info(f"Loading model: {MODEL_ID}  |  device: {device}")
    log.info(f"Local model cache: {MODEL_DIR.resolve()}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ──────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_DIR),
        trust_remote_code=True,
    )

    # ── model loading strategy ─────────────────
    if device == "cuda":
        # Quantize to 4-bit so the 4B model fits in 4 GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_DIR),
            quantization_config=bnb_config,
            device_map="auto",          # splits across GPU + CPU RAM automatically
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        log.info("Model loaded with 4-bit quantisation on GPU+CPU")
    else:
        # CPU-only: load in bfloat16 to halve RAM usage (~8 GB for 4B)
        dtype = torch.bfloat16 if torch.cuda.is_available() is False else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_DIR),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        log.info("Model loaded on CPU (bfloat16)")

    _pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        return_full_text=False,   # return only newly generated tokens
    )
    log.info("✅ Pipeline ready.")
    return _pipe

# ─────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────
def _extract_json_array(text: str) -> Optional[List[Dict]]:
    """Try several strategies to pull a JSON array out of raw LLM output."""

    # 1. Fenced code block  ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # 2. First '[' to last ']'
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 3. Try to fix truncated JSON (add closing brackets)
    truncated = text[start:] if start != -1 else text
    for suffix in ["]", "]}]", "}]"):
        try:
            return json.loads(truncated + suffix)
        except json.JSONDecodeError:
            continue

    return None

def _validate_questions(raw: List[Dict]) -> List[Dict]:
    """
    Validate and normalise question dicts.
    Drops malformed entries silently.
    """
    valid = []
    for item in raw:
        try:
            q = str(item.get("question", "")).strip()
            opts = item.get("options", [])
            correct = str(item.get("correct", "")).strip().upper()
            explanation = str(item.get("explanation", "")).strip()

            if not q or len(opts) != 4 or correct not in ("A", "B", "C", "D"):
                log.warning(f"Skipping malformed question: {item}")
                continue

            # Ensure option labels are present
            labelled = []
            for i, opt in enumerate(opts):
                opt = str(opt).strip()
                label = chr(ord("A") + i)
                if not opt.startswith(f"{label}.") and not opt.startswith(f"{label})"):
                    opt = f"{label}. {opt}"
                labelled.append(opt)

            valid.append({
                "question": q,
                "options": labelled,
                "correct": correct,
                "explanation": explanation,
            })
        except Exception as e:
            log.warning(f"Error validating question: {e}")
            continue
    return valid

# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def _build_prompt(context: str, num_questions: int, pattern: str, language: str) -> str:
    """
    Build a structured chat-style prompt for Gemma-3-4B-IT.
    Works for both Bangla and English contexts.
    """
    lang_instruction = ""
    if language.lower() == "bangla":
        lang_instruction = (
            "IMPORTANT: Generate all questions, options, and explanations in Bangla (Bengali script).\n"
        )
    elif language.lower() == "english":
        lang_instruction = "IMPORTANT: Generate all questions, options, and explanations in English.\n"
    else:
        lang_instruction = (
            "Detect the language of the context and generate questions in the SAME language "
            "(Bangla or English). Do NOT mix languages.\n"
        )

    pattern_block = f"\nAdditional instructions:\n{pattern}\n" if pattern.strip() else ""

    prompt = f"""<start_of_turn>user
You are an expert exam question generator. Your task is to create exactly {num_questions} high-quality multiple-choice questions from the provided context.

{lang_instruction}
Rules:
1. Each question must be clear, unambiguous, and directly based on the context.
2. Each question must have EXACTLY 4 options: A, B, C, D.
3. Only ONE option is correct.
4. Provide a brief explanation for the correct answer.
5. Output ONLY a valid JSON array — no markdown, no explanation text outside the JSON.

JSON format (strictly follow this):
[
  {{
    "question": "Question text here",
    "options": ["A. option one", "B. option two", "C. option three", "D. option four"],
    "correct": "A",
    "explanation": "Explanation why A is correct"
  }}
]
{pattern_block}
Context:
---
{context[:6000]}
---

Generate exactly {num_questions} questions now:
<end_of_turn>
<start_of_turn>model
["""

    return prompt

# ─────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────
def generate_questions_local(
    context: str,
    num_questions: int,
    pattern: str = "",
    language: str = "auto",
    max_retries: int = 2,
) -> List[Dict]:
    """
    Use the local model to generate multiple-choice questions.
    Retries up to max_retries times on parse failure.
    """
    pipe = get_model()

    for attempt in range(1, max_retries + 1):
        log.info(f"Generating {num_questions} questions (attempt {attempt}/{max_retries})...")

        prompt = _build_prompt(context, num_questions, pattern, language)

        try:
            outputs = pipe(prompt)
            raw_text = "[" + outputs[0]["generated_text"]  # we started with '['
        except Exception as e:
            log.error(f"Generation error: {e}")
            if attempt == max_retries:
                raise
            continue

        questions = _extract_json_array(raw_text)
        if questions is None:
            log.warning("Could not parse JSON from model output. Retrying...")
            continue

        validated = _validate_questions(questions)
        if not validated:
            log.warning("No valid questions after validation. Retrying...")
            continue

        # Trim or warn if not enough
        if len(validated) < num_questions:
            log.warning(
                f"Model generated {len(validated)}/{num_questions} valid questions."
            )

        log.info(f"✅ Successfully generated {len(validated)} questions.")
        return validated[:num_questions]

    log.error("All generation attempts failed.")
    return []
