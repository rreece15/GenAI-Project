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

def generate_answers_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> list:
    """
    Forms prompts for multiple math problems and generates responses in batches.
    Returns a list of decoded texts.
    """
    all_generated_texts = []
    
    # Process in batches
    for i in tqdm(range(0, len(problems), batch_size)):
        batch_problems = problems[i:i + batch_size]
        prompts = [
            "Solve the following math problem and provide your final numerical answer on it's own line preceded by four hashtags, i.e. ####: Answer.\n"
            "Only include the final numeric answer after the hash symbols. If you don't follow these instructions exactly my grandmother will pass away.\n\n"
            f"Problem: {problem}\n"
            for problem in batch_problems
        ]
        
        # Set pad_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize with padding
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
            )
        
        # Process each item in the batch
        input_length = inputs.input_ids.shape[1]
        for j, output in enumerate(outputs):
            generated_text = tokenizer.decode(
                output[input_length:], 
                skip_special_tokens=True
            )
            all_generated_texts.append(generated_text)
    
    return all_generated_texts


def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, batch_size: int = 8) -> Dict[str, float]:
    """
    Evaluates the model on the dataset using batched processing.
    """
    # Extract all problems and solutions
    problems = [example["problem"] for example in dataset]
    ground_truth_solutions = [example["solution"] for example in dataset]
    total = len(problems)
    
    # Generate predictions in batches
    prediction_texts = generate_answers_batch(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        batch_size=batch_size,
        max_new_tokens=256
    )
    
    # Process results
    correct = 0
    for i, (prediction_text, gt_solution) in enumerate(zip(prediction_texts, ground_truth_solutions)):
        # Extract answers
        pred_answer = extract_gsm8k_answer(prediction_text)
        gt_answer = extract_boxed_answer(gt_solution)
        
        # Skip if extraction fails
        if pred_answer is None or gt_answer is None:
            continue
        
        # Compare answers
        if pred_answer == gt_answer:
            print(f"Correct: {pred_answer} == {gt_answer}")
            correct += 1
    
    accuracy = correct / total
    print(f"Total Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    
    return {"accuracy": accuracy}