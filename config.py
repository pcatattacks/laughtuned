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
    "batch_size": 16,
    "gradient_accumulation_steps": 2,  # effective batch size 32 (matches spec exactly)
    "max_grad_norm": 1.0,
    "dpo_beta": 0.1,
    "kto_beta": 0.1,
    # Calibrated against the observed composite-score distribution from a
    # full judging pass (mean ≈ 2.64, judge runs harsh on Mistral base).
    # 3.5/2.5 produced ~8% desirable; lowering to 3.0 yields a more
    # balanced training set without re-judging.
    "kto_desirable_threshold": 3.0,
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
    # Forward-pass batch size for the BASE-model generation runs.
    # `gen_pairs_batch_size` is doubled internally (each prompt appears twice
    # in the forward pass to produce response_a and response_b from
    # independent samples), so the actual GPU batch is 2 * this value.
    # `gen_eval_batch_size` is used as-is for single-response generation
    # during final eval (no doubling).
    # Memory tuning rules of thumb on Mistral-7B 4-bit at seq_len=1024:
    #   T4 16 GB:        pairs=4,  eval=8
    #   A100 40 GB:      pairs=24, eval=48 (current, ~14 GB peak margin)
    #   A100 conservative: pairs=16, eval=32 (more headroom)
    "gen_pairs_batch_size": 24,
    "gen_eval_batch_size": 48,

    # APIs (fill in via Colab secrets or local env, do NOT commit values)
    "judge_model": "claude-sonnet-4-6",
    "guardian_api_key": "",
    "anthropic_api_key": "",

    # Guardian edition filter. One of "us", "uk", "aus", or None for global.
    # Note: "us" + a UK-default section name (e.g., "politics") returns
    # essentially nothing because the US team tags its content under
    # "us-news/*" paths, not the UK section names. Set to None to use
    # all eight spec sections; cherry-pick US examples for the report.
    "guardian_production_office": None,

    # Paths
    "drive_root": "/content/drive/MyDrive/Colab Notebooks/CS-5788-generative-models/final-project/",
}
