import json
import os

import evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
    batch_size=8,
    k=10,
    max_new_tokens=256,
    temperature=0.2,
    save_path=None,
    save_every=5,  # Save after processing every 5 samples
):
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resize embeddings if needed
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    candidates = []
    prompts = []
    meta_info = []

    for i, sample in enumerate(tqdm(dataset, desc="Generating candidates")):
        # Collect the prompts and associated metadata
        prompts.append(sample["prompt"])
        meta_info.append(
            {
                "task_id": f"humaneval_{i}",
                "prompt": sample["prompt"],
                "entry_point": sample["entry_point"],
                "test": sample["test"],
            }
        )

        # Process batch when enough samples are collected or at end of dataset
        if len(prompts) == batch_size or i == len(dataset) - 1:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=k,
                )
            # Decode and split outputs per prompt; output_ids is [batch_size * k, ...]
            decoded_outputs = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            for j in range(len(prompts)):
                candidate_completions = decoded_outputs[j * k : (j + 1) * k]
                meta_info[j]["candidates"] = candidate_completions
                candidates.append(meta_info[j])

            # Reset for next batch
            prompts, meta_info = [], []

            # Periodically save candidates if a file path is provided
            if save_path and len(candidates) % save_every < batch_size:
                with open(save_path, "w") as f:
                    json.dump(candidates, f)
                print(
                    f"Candidates saved to {save_path} after processing {i + 1} samples."
                )

    # Final save in case there are unsaved candidates
    if save_path:
        with open(save_path, "w") as f:
            json.dump(candidates, f)
        print(f"Final candidates saved to {save_path}")

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
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
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
