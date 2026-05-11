"""Generic preference-optimization training loop for DPO and KTO.

The same loop drives both losses — the only thing that varies is the
``loss_fn`` callback and the shape of the batch dict it consumes. The
loop is responsible for:

- AdamW optimizer + cosine LR schedule with warmup.
- Gradient accumulation (``config["gradient_accumulation_steps"]``).
- Gradient clipping to ``config["max_grad_norm"]``.
- Periodic validation every ``config["eval_steps"]`` optimizer steps.
- Checkpointing LoRA adapter weights every ``config["save_steps"]`` steps.
- Early stopping on ``config["early_stopping_patience"]`` evals without
  improvement; restores the best adapter weights on exit.
- Streaming all per-step metrics through ``logger`` (TensorBoard + JSONL).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_scheduler

from models.log_probs import compute_log_probs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LossFn = Callable[..., Tuple[torch.Tensor, Dict[str, float]]]


# ---------------------------------------------------------------------------
# Per-batch dispatch: DPO vs KTO
# ---------------------------------------------------------------------------


def _is_dpo_batch(batch: Dict[str, torch.Tensor]) -> bool:
    """DPO batches carry chosen/rejected pairs; KTO batches don't."""
    return "chosen_input_ids" in batch


def _forward_one_batch(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    ref_log_probs: Dict[str, torch.Tensor] | torch.Tensor,
    loss_fn: LossFn,
    beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one batch through the policy + reference lookup + loss.

    Dispatches by batch shape so the caller doesn't need to know which
    algorithm is running.
    """
    idx = batch["example_idx"]

    if _is_dpo_batch(batch):
        assert isinstance(ref_log_probs, dict), "DPO needs a dict of ref tensors"
        policy_chosen_logps_B = compute_log_probs(
            model,
            batch["chosen_input_ids"].to(DEVICE),
            batch["chosen_attention_mask"].to(DEVICE),
            batch["chosen_label_mask"].to(DEVICE),
        )
        policy_rejected_logps_B = compute_log_probs(
            model,
            batch["rejected_input_ids"].to(DEVICE),
            batch["rejected_attention_mask"].to(DEVICE),
            batch["rejected_label_mask"].to(DEVICE),
        )
        # precomputed reference policy log-probs
        reference_chosen_logps_B = ref_log_probs["chosen"][idx].to(DEVICE)
        reference_rejected_logps_B = ref_log_probs["rejected"][idx].to(DEVICE)
        return loss_fn(
            policy_chosen_logps_B,
            policy_rejected_logps_B,
            reference_chosen_logps_B,
            reference_rejected_logps_B,
            beta,
        )

    # KTO branch
    assert isinstance(ref_log_probs, torch.Tensor), "KTO needs a flat ref tensor"
    policy_logps_B = compute_log_probs(
        model,
        batch["input_ids"].to(DEVICE),
        batch["attention_mask"].to(DEVICE),
        batch["label_mask"].to(DEVICE),
    )
    reference_logps_B = ref_log_probs[idx].to(DEVICE)
    labels_B = batch["label"].to(DEVICE)
    return loss_fn(policy_logps_B, reference_logps_B, labels_B, beta)


@torch.inference_mode()
def _evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader[Dict[str, torch.Tensor]],
    ref_log_probs: Dict[str, torch.Tensor] | torch.Tensor,
    loss_fn: LossFn,
    beta: float,
) -> Dict[str, float]:
    """Run a full validation pass; return per-key mean metrics."""
    was_training = model.training
    model.eval()

    aggregated: Dict[str, float] = {}
    n_batches = 0
    for batch in dataloader:
        _, metrics = _forward_one_batch(
            model, batch, ref_log_probs, loss_fn, beta
        )
        for k, v in metrics.items():
            aggregated[k] = aggregated.get(k, 0.0) + float(v)
        n_batches += 1

    if was_training:
        model.train()

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in aggregated.items()}


# ---------------------------------------------------------------------------
# Resume-from-checkpoint
# ---------------------------------------------------------------------------


def _load_resume_state(
    checkpointer: Any,
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    experiment_name: str,
) -> Dict[str, Any]:
    """Restore training state from disk if a checkpoint exists.

    Reloads the LoRA adapter weights onto ``model`` and the optimizer +
    scheduler state in place. Returns a dict of bookkeeping counters
    (step / epoch / best_val_loss / best_step / evals_without_improvement /
    last_val_loss). If no checkpoint is present or the load fails, the
    returned dict carries fresh defaults so the caller can use it
    unconditionally.
    """
    fresh: Dict[str, Any] = {
        "global_step": 0,
        "start_epoch": 0,
        "best_val_loss": float("inf"),
        "best_step": 0,
        "evals_without_improvement": 0,
        "last_val_loss": float("inf"),
    }
    if not checkpointer.has_resume_point():
        return fresh

    try:
        print(
            f"[train/{experiment_name}] resuming from checkpoint at "
            f"{checkpointer._latest_path}"
        )
        checkpointer.load_latest_adapter(model)
        state = checkpointer.load_latest(optimizer, scheduler)
        out: Dict[str, Any] = {
            "global_step": int(state.get("step", 0)),
            "start_epoch": int(state.get("epoch", 0)),
            "best_val_loss": float(state.get("best_val_loss", float("inf"))),
            "best_step": int(state.get("best_step", 0)),
            "evals_without_improvement": int(
                state.get("evals_without_improvement", 0)
            ),
            "last_val_loss": float(state.get("val_loss", float("inf"))),
        }
        print(
            f"[train/{experiment_name}] resumed at epoch={out['start_epoch']}, "
            f"step={out['global_step']}, "
            f"best_val_loss={out['best_val_loss']:.4f}@step{out['best_step']}"
        )
        return out
    except Exception as e:
        print(f"[train/{experiment_name}] resume failed ({e}); starting fresh")
        return fresh


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    model: PreTrainedModel,
    train_dataloader: DataLoader[Dict[str, torch.Tensor]],
    val_dataloader: DataLoader[Dict[str, torch.Tensor]],
    ref_log_probs_train: Dict[str, torch.Tensor] | torch.Tensor,
    ref_log_probs_val: Dict[str, torch.Tensor] | torch.Tensor,
    loss_fn: LossFn,
    config: Dict[str, Any],
    experiment_name: str,
    logger: Any,
    checkpointer: Any,
) -> Dict[str, Any]:
    """Run the full training loop with validation, checkpointing, early stopping.

    Args:
        model: The LoRA-wrapped policy model. Only adapter weights have
            ``requires_grad=True``.
        train_dataloader: DataLoader over the training preference data.
            Each batch is a dict of tensors; the exact keys depend on the
            algorithm (DPO has chosen/rejected pairs; KTO has single
            responses with binary labels). Each batch must include
            ``example_idx`` so we can look up cached reference log-probs.
        val_dataloader: DataLoader over the validation preference data,
            same shape as ``train_dataloader``.
        ref_log_probs_train: Precomputed reference log-probs for the
            training set. For DPO: dict with keys ``"chosen"`` and
            ``"rejected"`` (each ``(N_train,)``). For KTO: a single
            ``(N_train,)`` tensor.
        ref_log_probs_val: Same shape as ``ref_log_probs_train``, indexed
            against the validation set.
        loss_fn: Either ``dpo_loss`` or ``kto_loss``. Must accept the
            appropriate per-batch policy/reference log-probs and return
            ``(loss_scalar, metrics_dict)``.
        config: Full project config dictionary. Reads ``learning_rate``,
            ``lr_scheduler``, ``warmup_ratio``, ``num_epochs``,
            ``batch_size``, ``gradient_accumulation_steps``,
            ``max_grad_norm``, ``dpo_beta`` / ``kto_beta``, ``eval_steps``,
            ``save_steps``, ``early_stopping_patience``.
        experiment_name: Unique tag (e.g. ``"dpo_rubric_short"``) used by
            ``logger`` and ``checkpointer`` to namespace outputs on Drive.
        logger: Object exposing ``log_step(metrics, step)`` and
            ``log_eval(metrics, step)``. TensorBoard + JSONL writer.
        checkpointer: Object exposing ``save(model, optimizer, scheduler,
            step, val_loss, best_val_loss)`` and ``load_best(model)``.

    Returns:
        Summary dict with at least:
            - ``best_val_loss``: float
            - ``best_step``: int
            - ``total_steps``: int
            - ``training_time_sec``: float
            - ``final_train_loss``: float
    """
    # --- Hyperparameters --------------------------------------------------
    num_epochs = int(config["num_epochs"])
    grad_accum = max(1, int(config["gradient_accumulation_steps"]))
    max_grad_norm = float(config["max_grad_norm"])
    eval_steps = max(1, int(config["eval_steps"]))
    save_steps = max(1, int(config["save_steps"]))
    patience = max(1, int(config["early_stopping_patience"]))

    fn_name = getattr(loss_fn, "__name__", "")
    beta = float(config["kto_beta"] if fn_name == "kto_loss" else config["dpo_beta"])

    # --- Optimizer + scheduler --------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(config["learning_rate"]))

    micro_per_epoch = len(train_dataloader)
    total_opt_steps = max(1, (micro_per_epoch // grad_accum) * num_epochs)
    warmup_steps = int(float(config["warmup_ratio"]) * total_opt_steps)
    scheduler = get_scheduler(
        str(config["lr_scheduler"]),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    # --- State (fresh defaults, optionally overwritten by resume) --------
    resume_state = _load_resume_state(
        checkpointer, model, optimizer, scheduler, experiment_name
    )
    global_step: int = resume_state["global_step"]
    start_epoch: int = resume_state["start_epoch"]
    best_val_loss: float = resume_state["best_val_loss"]
    best_step: int = resume_state["best_step"]
    evals_without_improvement: int = resume_state["evals_without_improvement"]
    last_val_loss: float = resume_state["last_val_loss"]
    final_train_loss: float = float("nan")
    accumulated_metrics: Dict[str, float] = {}
    start_time = time.time()
    early_stop = False

    print(
        f"[train/{experiment_name}] start | epochs={start_epoch}->{num_epochs} "
        f"| micro/epoch={micro_per_epoch} | grad_accum={grad_accum} "
        f"| opt_steps_total={total_opt_steps} | warmup={warmup_steps} "
        f"| beta={beta} | lr={config['learning_rate']}"
    )

    optimizer.zero_grad()

    # --- Main training loop ----------------------------------------------
    for epoch in tqdm(
        range(start_epoch, num_epochs),
        desc=f"{experiment_name} epochs",
        initial=start_epoch,
        total=num_epochs,
        leave=True,
    ):
        if early_stop:
            break
        model.train()
        for batch_idx, batch in tqdm(
            enumerate(train_dataloader),
            total=micro_per_epoch,
            desc=f"epoch {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            # 1) Forward + loss for this microbatch
            loss, metrics = _forward_one_batch(
                model, batch, ref_log_probs_train, loss_fn, beta
            )

            # 2) Scaled backward (so accumulated grads average over microbatches)
            (loss / grad_accum).backward()

            for k, v in metrics.items():
                accumulated_metrics[k] = (
                    accumulated_metrics.get(k, 0.0) + float(v) / grad_accum
                )

            # 3) Optimizer step every `grad_accum` microbatches
            if (batch_idx + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 4) Log step
                lr = scheduler.get_last_lr()[0]
                vram_gb = (
                    torch.cuda.memory_allocated() / 1e9
                    if torch.cuda.is_available()
                    else 0.0
                )
                step_metrics = {
                    **accumulated_metrics,
                    "lr": lr,
                    "grad_norm": float(grad_norm.item()),
                    "vram_gb": vram_gb,
                }
                logger.log_step(step_metrics, global_step)
                final_train_loss = accumulated_metrics.get(
                    "loss", final_train_loss
                )
                accumulated_metrics = {}

                # 5) Periodic validation
                if global_step % eval_steps == 0:
                    val_metrics = _evaluate(
                        model, val_dataloader, ref_log_probs_val,
                        loss_fn, beta,
                    )
                    logger.log_eval(val_metrics, global_step)
                    last_val_loss = val_metrics.get("loss", last_val_loss)
                    if last_val_loss < best_val_loss:
                        best_val_loss = last_val_loss
                        best_step = global_step
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1

                # 6) Periodic checkpoint
                if global_step % save_steps == 0:
                    checkpointer.save(
                        model, optimizer, scheduler,
                        step=global_step,
                        epoch=epoch,
                        val_loss=last_val_loss,
                        best_val_loss=best_val_loss,
                        best_step=best_step,
                        evals_without_improvement=evals_without_improvement,
                    )

                # 7) Early stopping
                if evals_without_improvement >= patience:
                    print(
                        f"[train/{experiment_name}] early stop at step "
                        f"{global_step}: {patience} evals without improvement"
                    )
                    early_stop = True
                    break

    # --- Final eval + save + restore best --------------------------------
    if global_step > 0:
        final_val_metrics = _evaluate(
            model, val_dataloader, ref_log_probs_val, loss_fn, beta,
        )
        logger.log_eval(final_val_metrics, global_step)
        final_val_loss = final_val_metrics.get("loss", last_val_loss)
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_step = global_step
        # Save final state with epoch=num_epochs so a future resume sees
        # "already done" (empty outer-loop range) and skips straight to
        # the final eval + load_best step.
        checkpointer.save(
            model, optimizer, scheduler,
            step=global_step,
            epoch=num_epochs,
            val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_step=best_step,
            evals_without_improvement=evals_without_improvement,
        )
        try:
            checkpointer.load_best(model)
        except Exception as e:
            print(f"[train/{experiment_name}] load_best failed: {e}")

    elapsed = time.time() - start_time
    print(
        f"[train/{experiment_name}] done | best_val_loss={best_val_loss:.4f} "
        f"@step {best_step} | total_steps={global_step} | time={elapsed:.1f}s"
    )
    return {
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "total_steps": global_step,
        "training_time_sec": elapsed,
        "final_train_loss": final_train_loss,
    }
