"""
model_utils.py — Shared checkpoint helpers for the multi-generation pipeline.

A generation's student can be stored in two formats depending on --run:
  * LoRA adapter  (adam_lora / sgd_lora)  — needs the base model + merge
  * full model    (full_ft)               — a complete checkpoint, load directly

Both can live on the HF Hub or in a local directory (full_ft --no-hub).

`is_peft_checkpoint` inspects the checkpoint BEFORE any model is instantiated, so
a full-parameter student never triggers a wasted ~14 GB base-model load, and
`load_student` never silently falls back to the unsteered base model.
"""

import os


def is_peft_checkpoint(path_or_repo: str) -> bool:
    """True if `path_or_repo` is a PEFT/LoRA adapter, False if it is a full model.

    Local directory: decided by the presence of adapter_config.json.
    Hub repo id:     decided by whether a PeftConfig can be fetched.
    """
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, "adapter_config.json"))
    try:
        from peft import PeftConfig
        PeftConfig.from_pretrained(path_or_repo)
        return True
    except Exception:
        return False


def load_student(base_model: str, checkpoint=None, **from_pretrained_kwargs):
    """Load a (merged) model for inference.

    checkpoint=None          -> the plain base model (generation 1 teacher / base eval)
    checkpoint = LoRA adapter -> base model + adapter, merged in place
    checkpoint = full model   -> loaded directly; `base_model` is NOT loaded

    Raises on failure — never falls back to the base model.
    """
    from transformers import AutoModelForCausalLM

    if checkpoint is None:
        return AutoModelForCausalLM.from_pretrained(base_model, **from_pretrained_kwargs)

    if is_peft_checkpoint(checkpoint):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base_model, **from_pretrained_kwargs)
        return PeftModel.from_pretrained(model, checkpoint).merge_and_unload()

    return AutoModelForCausalLM.from_pretrained(checkpoint, **from_pretrained_kwargs)
