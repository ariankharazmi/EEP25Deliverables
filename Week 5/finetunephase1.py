import torch  # PyTorch for training purposes
from accelerate import Accelerator  # Need this to address numpy issues
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling)  # OpenAI and HuggingFace packages for training, training configs, and data batch preparation
from datasets import load_dataset, concatenate_datasets  # Hugging Face datasets
import json
import pandas as pd
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()
print("Emissions tracker has started.")
print("Beginning transfer learning (phase 1 fine-tuning)...")

# Load the tokenizer and GPT-2 model for use. GPT-2 allows for local training/fine-tuning and no API key, so this is perfect for a student project.
transformer_model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
model = AutoModelForCausalLM.from_pretrained(transformer_model_name)

# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (from EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True

open_ds = load_dataset("OpenAssistant/oasst1", split='train[:100%]', trust_remote_code=True) ## streaming=True)
bbc_ds = load_dataset("permutans/fineweb-bbc-news", "CC-MAIN-2022-33", split='train[:100%]', trust_remote_code=True) ## streaming=True)
wiki_ds = load_dataset("NeuML/wikipedia-20250620", split='train[:25000]', trust_remote_code=True) ## streaming=True)
essen_ds = load_dataset("EssentialAI/essential-web-v1.0", "CC-MAIN-2024-38", split='train[:25000]', trust_remote_code=True) ## streaming=True)
alpaca_ds = load_dataset("yahma/alpaca-cleaned", split='train[:100%]', trust_remote_code=True) ## streaming=True) ## Moved Alpaca Data to Phase I from Phase II
squad_ds = load_dataset("rajpurkar/squad_v2", split='train[:100%]', trust_remote_code=True) ## streaming=True)
eli_ds = load_dataset("sentence-transformers/eli5", split='train[:100%]', trust_remote_code=True) ## streaming=True)
dolly_ds = load_dataset("llm-wizard/dolly-15k-instruction-alpaca-format", split='train[:100%]', trust_remote_code=True) ## streaming=True)

phase1 = concatenate_datasets([open_ds, bbc_ds, wiki_ds, essen_ds, alpaca_ds, squad_ds, eli_ds, dolly_ds])

phase1 = phase1.map(preprocess_function, batched=True, remove_columns=phase1.column_names)
phase1.set_format("torch", ["input_ids","attention_mask"])

def preprocess_function(dataset_column_examples):  # Looks for examples as input, used as a dictionary
    # Text fields can be adjusted based on data columns for dataset(s)
    text_fields = [
        'text', 'prompt', ' response', 'chosen', 'rejected', 'content',
        'sentence', 'concept_name', 'context',
        'column', 'id', 'name', 'instruction', 'instances',
        'input', 'noinput', 'output']  # Adjusted for dataset(s) columns, looks for keywords in examples dictionary
    for field in text_fields:  # Goes through list of text fields (loops), if field exists it assigns value to texts and leaves loop
        if field in dataset_column_examples:
            texts = dataset_column_examples[field]
            break
    else:
        raise ValueError(f"No available text fields were found: {dataset_column_examples.keys()}")  # If no assigned values are found, the program breaks
    # Elements MUST be strings (or process will break)
    texts = [str(text) if text is not None else "" for text in texts]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=512)  # Adjust if needed -- uniformity in sequence tokenization, longer sequences are truncated