"""Central configuration for the LaughTuned project.

Every hyperparameter and path lives here. Nothing should be hardcoded
elsewhere in the code repo.
"""

from typing import Any, Dict

CONFIG: Dict[str, Any] = {
    # Reproducibility
    "seed": 42,

    # Model
    "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
    "max_seq_length": 1024,

    # QLoRA
    "quantization_bits": 4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

    # Training
    "learning_rate": 5e-5,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,  # effective batch size 32
    "max_grad_norm": 1.0,
    "dpo_beta": 0.1,
    "kto_beta": 0.1,
    "kto_desirable_threshold": 3.5,
    "kto_undesirable_threshold": 2.5,

    # Evaluation and checkpointing
    "eval_steps": 50,
    "save_steps": 100,
    "early_stopping_patience": 3,  # in units of eval_steps

    # Data
    "num_comedy_prompts": 500,
    "num_eval_prompts": 30,
    "val_ratio": 0.1,

    # Generation
    "gen_temperature": 0.9,
    "gen_top_p": 0.95,
    "gen_max_new_tokens": 200,

    # APIs (fill in via Colab secrets or local env, do NOT commit values)
    "judge_model": "claude-sonnet-4-6",
    "guardian_api_key": "",
    "anthropic_api_key": "",

    # Guardian edition filter. One of "us", "uk", "aus", or None for global.
    "guardian_production_office": "us",

    # Paths
    "drive_root": "/content/drive/MyDrive/Colab Notebooks/CS-5788-generative-models/final-project/",
}
