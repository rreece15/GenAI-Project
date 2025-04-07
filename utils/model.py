from typing import Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_peft_model(
    model_id: str,
    peft_model_id: str,
    adapter_name: str = "default",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a PEFT model and tokenizer from Hugging Face Hub.

    Args:
        model_id (str): The base model ID.
        peft_model_id (str): The PEFT model ID.
        device_map (str, optional): Device map for loading the model. Defaults to "auto".
        torch_dtype (torch.dtype, optional): Torch data type for the model. Defaults to torch.float16.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    peft_model = PeftModel.from_pretrained(model, peft_model_id, adapter_name=adapter_name, device_map=device_map)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        peft_model.config.pad_token = tokenizer.eos_token
        peft_model.config.pad_token_id = tokenizer.eos_token_id

    return peft_model, tokenizer
