from typing import Dict, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_peft_model(
    model_id: str,
    adapter_ids: Dict[str, str],
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model and multiple adapter checkpoints from Hugging Face Hub,
    then combine them by averaging their weights equally using add_weighted_adapter.

    Args:
        model_id (str): The base model ID.
        adapter_ids (Dict[str, str]): A dictionary mapping adapter names to their checkpoint IDs.
        device_map (str, optional): Device map for model loading. Defaults to "auto".
        torch_dtype (torch.dtype, optional): Torch data type. Defaults to torch.float16.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The PEFT model with the merged adapter and the tokenizer.
    """
    # Load the base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Get list of adapter names in the dictionary and select the first as the primary adapter.
    adapter_names = list(adapter_ids.keys())
    if not adapter_names:
        raise ValueError("No adapters provided in adapter_ids.")

    primary_adapter_name = adapter_names[0]
    primary_adapter_checkpoint = adapter_ids[primary_adapter_name]

    # Load the primary adapter
    model = PeftModel.from_pretrained(
        base_model,
        primary_adapter_checkpoint,
        adapter_name=primary_adapter_name,
        device_map=device_map,
    )

    # Load any additional adapters
    for adapter_name in adapter_names[1:]:
        adapter_checkpoint = adapter_ids[adapter_name]
        model.load_adapter(adapter_checkpoint, adapter_name=adapter_name)

    # Compute equal weights for each adapter (averaging)
    num_adapters = len(adapter_names)
    weights = [1.0 / num_adapters] * num_adapters

    # Add a weighted (linear) merged adapter from the loaded adapters
    model.add_weighted_adapter(adapters=adapter_names, weights=weights, adapter_name="merged", combination_type="linear")
    model.set_adapter("merged")

    # Fix pad token if necessary.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
