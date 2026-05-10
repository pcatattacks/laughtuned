"""QLoRA model loading for LaughTuned.

Loads Mistral-7B-Instruct-v0.2 in 4-bit NF4 with double quantization, then
attaches LoRA adapters on the attention projections. Returns the wrapped
``PeftModel`` ready for training (and the matching tokenizer).

VRAM expectations on Colab:
- 4-bit base weights: ~4 GB
- Activations + KV cache during forward: ~6-10 GB at seq_len=1024, bs=4
- LoRA adapters + optimizer state (rank 16): <1 GB
- Total at training time: comfortably fits T4 (16 GB) for seq_len=1024.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    config: Dict[str, Any],
) -> Tuple[PeftModel, PreTrainedTokenizerBase]:
    """Load the base model in 4-bit NF4 and wrap it in LoRA adapters.

    Args:
        config: Project config dict. Must contain ``base_model``,
            ``lora_rank``, ``lora_alpha``, ``lora_dropout``, and
            ``lora_target_modules``.

    Returns:
        A tuple ``(model, tokenizer)``. ``model`` is a ``PeftModel`` whose
        only trainable parameters are the freshly initialized LoRA
        adapters; the underlying base weights are frozen and quantized.
        ``tokenizer`` has ``pad_token`` set (falling back to ``eos_token``
        when the model has no native pad token).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model_name: str = config["base_model"]
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["lora_target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model: PeftModel = get_peft_model(base_model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _print_param_summary(model)
    _print_vram_usage()

    return model, tokenizer


def _print_param_summary(model: PeftModel) -> None:
    """Print trainable vs total parameter counts for the LoRA-wrapped model."""
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    pct = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"[load_model] Trainable params: {trainable:,} / {total:,} "
        f"({pct:.3f}%)"
    )


def _print_vram_usage() -> None:
    """Print current GPU memory usage. No-op when CUDA is unavailable."""
    if not torch.cuda.is_available():
        print("[load_model] CUDA not available; skipping VRAM report.")
        return
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    device_name = torch.cuda.get_device_name(0)
    print(
        f"[load_model] GPU: {device_name} | "
        f"allocated: {allocated_gb:.2f} GB | reserved: {reserved_gb:.2f} GB"
    )
