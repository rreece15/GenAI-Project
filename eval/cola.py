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

def generate_answers_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: list,
    batch_size: int = 8,
    max_new_tokens: int = 4,
) -> list:
    """
    Forms prompts for multiple sentences and generates responses in batches.
    Returns a list of decoded texts.
    """
    all_generated_texts = []
    
    # Process in batches
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i + batch_size]
        prompts = [
            'Determine if the sentence below is syntactically and semantically correct. If it is syntactically and semantically correct, respond "1". Otherwise, respond "0". '
            f"\nSentence: {sentence}\n"
            "Answer:"
            for sentence in batch_sentences
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


def evaluate(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    dataset: Dataset,
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Evaluates the model on the dataset using batched processing.
    """
    # Extract all sentences and labels
    sentences = [example["sentence"] for example in dataset]
    ground_truths = [example["label"] for example in dataset]
    
    # Generate predictions in batches
    prediction_texts = generate_answers_batch(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        batch_size=batch_size,
        max_new_tokens=256
    )
    
    # Extract labels from predictions
    predictions = [extract_label(text) for text in prediction_texts]
    
    # Calculate accuracy
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    accuracy = correct / len(ground_truths)
    
    return {"accuracy": accuracy}