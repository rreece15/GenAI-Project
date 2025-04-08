import re
from typing import Dict

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_boxed_answer(text):
    """
    Extracts the content of the last encountered \boxed{...} instance from a given string.
    Returns None if no such instance is found.
    """
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    return matches[-1].strip() if matches else None


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final answer from the text output, where the answer is expected '####'.
    Returns None if no such line is found.
    """
    # Use multi-line mode (?m) to search for the line starting with ####.
    match = re.search(r"(?m)^####\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip()
    return None


def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, problem_text: str, max_new_tokens: int = 256) -> str:
    """
    Forms a prompt for the given problem, generates a response with the model,
    and returns the decoded text.
    """
    prompt = (
        "Solve the following math problem and provide your final numerical answer on it's own line preceded by four hashtags, i.e. ####: Answer.\n"
        "Only include the final numeric answer after the hash symbols. If you don't follow these instructions exactly my grandmother will pass away.\n\n"
        f"Problem: {problem_text}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=True)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)

    return generated_text


def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset) -> Dict[str, float]:
    correct = 0
    total = len(dataset)

    # Iterate over all examples in the dataset
    for example in tqdm(dataset, total=total):
        # Each example should have the fields: 'problem', 'solution'
        problem = example["problem"]
        ground_truth_solution = example["solution"]

        # Generate the prediction from the model for the current problem.
        prediction_text = generate_answer(model, tokenizer, problem)

        # Extract the final boxed answer from both the model output and ground truth solution.
        pred_answer = extract_gsm8k_answer(prediction_text)
        gt_answer = extract_boxed_answer(ground_truth_solution)

        # If extraction fails for either, consider that example incorrect.

        if pred_answer is None or gt_answer is None:
            continue

        # Compare the extracted answers
        if pred_answer == gt_answer:
            print(f"Correct: {pred_answer} == {gt_answer}")
            correct += 1

    accuracy = correct / total

    print(f"Total Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
