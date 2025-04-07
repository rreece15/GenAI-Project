import json
import os

import evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def save_candidates(candidates, file_path):
    with open(file_path, "w") as f:
        json.dump(candidates, f)
    print(f"Candidates saved to {file_path}")


def load_candidates(file_path):
    with open(file_path, "r") as f:
        candidates = json.load(f)
    print(f"Candidates loaded from {file_path}")
    return candidates


def generate_candidates(
    model,
    tokenizer,
    dataset,
    k=10,
    max_new_tokens=256,
    temperature=0.2,
    save_path=None,
):
    """
    Generates k candidate completions for each sample in the dataset using batched generation.

    Each sample is expected to be a dict with keys "prompt", "entry_point", and "test".

    Args:
        model: The language model to use for generation.
        tokenizer: The corresponding tokenizer.
        dataset: The dataset containing HumanEval examples.
        k (int): Number of completions to generate per prompt.
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        save_path (str): Optional path to save the candidates.

    Returns:
        list: A list of candidate dictionaries.
    """
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resize model embeddings if new tokens were added
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    candidates = []
    for i, sample in enumerate(tqdm(dataset, desc="Generating candidates")):
        # Ensure the sample is a dict with the expected keys
        if not isinstance(sample, dict):
            raise ValueError(
                f"Expected sample to be a dict, got {type(sample)}: {sample}"
            )
        prompt = sample["prompt"]
        entry_point = sample["entry_point"]
        test = sample["test"]

        # Batch generation: generate k completions in one call
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=k,  # Generates k completions in one batch call
            )
        candidate_completions = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        candidates.append(
            {
                "task_id": f"humaneval_{i}",
                "prompt": prompt,
                "entry_point": entry_point,
                "test": test,
                "candidates": candidate_completions,
            }
        )
    if save_path:
        with open(save_path, "w") as f:
            json.dump(candidates, f)
        print(f"Candidates saved to {save_path}")
    return candidates


def evaluate_candidates(
    candidates,
    k=10,
):
    # Load the model and tokenizer

    code_eval_metric = evaluate.load("code_eval")
    pass_at_k, results = code_eval_metric.compute(
        predictions=[c["candidates"] for c in candidates],
        references=[c["test"] for c in candidates],
        k=[1, k],
        num_workers=4,
    )

    print("Evaluation results:")
    for k in [1, k]:
        print(f"Pass@{k}: {pass_at_k[f'pass@{k}'] * 100:.2f}%")


if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-v0.1"
    candidate_file = "data/generated_results/candidates.json"
    k = 5
    max_new_tokens = 256
    temperature = 0.2
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_from_disk("data/humaneval_dataset")["test"]

    generate_candidates(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        save_path=candidate_file,
    )

    candidates = load_candidates(candidate_file)
    evaluate_candidates(candidates, k=k)
