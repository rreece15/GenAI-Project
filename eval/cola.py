import re
from typing import Dict

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_label(s: str) -> int:
    """
    Extracts the first occurrence of '0' or '1' in the given string.
    Returns:
        0 or 1 as an integer.
    Raises:
        ValueError: If neither '0' nor '1' is found in the string.
    """
    match = re.search(r"[01]", s)
    if match:
        return int(match.group())
    else:
        return -1


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem_text: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Forms a prompt for the given problem, generates a response with the model,
    and returns the decoded text.
    """
    prompt = f"""Determine if the sentence below is syntactically and semantically correct. If it is syntactically and semantically correct, respond "1". Otherwise, respond "0". Only include the final numerical answer preceded by four hashtags, i.e. ####: Answer.\n If you don't follow these instructions exactly my grandmother will pass away.\n\nSentence: {problem_text}\n"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
    )

    return generated_text


def evaluate(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset
) -> Dict[str, float]:
    correct = 0
    total = len(dataset)

    for example in tqdm(dataset, total=total):
        problem_text = example["sentence"]
        ground_truth = example["label"]

        prediction_text = generate_answer(model, tokenizer, problem_text)

        prediction = int(extract_label(prediction_text))
        if prediction == ground_truth:
            correct += 1
    accuracy = correct / total
    return {"accuracy": accuracy}
