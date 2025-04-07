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


def generate_answer(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, problem_text: str, max_new_tokens: int = 256) -> str:
    """
    Forms a prompt for the given problem, generates a response with the model,
    and returns the decoded text.
    """
    prompt = "Answer the following math problem and provide your final answer enclosed in \\boxed{}:\n\n" f"Problem: {problem_text}\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    input_ids = inputs.input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

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
        pred_answer = extract_boxed_answer(prediction_text)
        gt_answer = extract_boxed_answer(ground_truth_solution)

        # If extraction fails for either, consider that example incorrect.
        if pred_answer is None or gt_answer is None:
            continue

        # Compare the extracted answers
        if pred_answer == gt_answer:
            correct += 1

    accuracy = correct / total

    print(f"Total Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
