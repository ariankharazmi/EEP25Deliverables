import torch  # PyTorch for training purposes
from accelerate import Accelerator  # Need this to address numpy issues
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling  # OpenAI and HuggingFace packages for training, training configs, and data batch preparation
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

# Set the padding token to the EOS token -- This keeps tokenization for sequences consistent, keeping attention masking simplified.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True

# Local dataset (seed_tasks.jsonl)
#json_ds = load_dataset('json', data_files='seed_tasks_2MB.jsonl', split='train')

# Load datasets from Hugging Face.
# Working datasets

# New Curiosity-16 Datasets
alpaca_ds = load_dataset("tatsu-lab/alpaca")

# Datasets from Curiosity 15
open_ds = load_dataset("OpenAssistant/oasst1", split='train[:100%]', trust_remote_code=True)
comb_ds = load_dataset("yoonholee/combined-preference-dataset", split='train[:100%]', trust_remote_code=True)
pref_ds = load_dataset("OpenRLHF/preference_dataset_mixture2_and_safe_pku", split='train[:100%]', trust_remote_code=True)
com_ds = load_dataset("community-datasets/generics_kb", "generics_kb_simplewiki", split='train[:100%]', trust_remote_code=True)

# List of datasets that do not work in conjunction with each other.
# congpt_ds = load_dataset("routellm/gpt4_dataset", split='train[:5%]', trust_remote_code=True)
# reward_ds = load_dataset("allenai/reward-bench", split='filtered[:5%]', trust_remote_code=True)

# Combine dataset(s), make sure your datasets are compatible with each other
combined_dataset = concatenate_datasets([open_ds, comb_ds, pref_ds, com_ds])

# Preprocess function for the combined dataset(s)
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
    return tokenizer(texts, truncation=True, padding='max_length', max_length=256)  # Adjust if needed -- uniformity in sequence tokenization, longer sequences are truncated

# Print dataset (column) information (also good for debugging when your combined dataset(s) don't work together)
print("Dataset columns:", combined_dataset.column_names)
print("Sample data from datasets:")
print(combined_dataset[:5])

# Tokenize the combined dataset(s)
tokenized_datasets = combined_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])

# Finding (len) size of dataset(s) for future partitioning (breaking into smaller sets)
dataset_size = len(tokenized_datasets)

# Define the size of the subsets, for training sets and eval sets, good for setting sizes later
# train_size = min(1000, dataset_size)
# eval_size = min(200, dataset_size)
# test_size = min(200, dataset_size)

# Shuffle and split the dataset
# shuffled_dataset = tokenized_datasets.shuffle(seed=42)
# mall_train_dataset = shuffled_dataset.select(range(train_size))
# small_eval_dataset = shuffled_dataset.select(range(train_size, train_size + eval_size))
# small_test_dataset = shuffled_dataset.select(range(train_size + eval_size, train_size + eval_size + test_size))

# Define training args
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Smaller batch size for faster processing speeds/time
    per_device_eval_batch_size=2,  # Smaller batch size for faster processing speeds/time
    num_train_epochs=3,  # Increase number of epochs (cycles of running through)
    weight_decay=0.01,
    save_total_limit=2,  # Number of checkpoints that will be saved to the filePATH
)

# Data collator function (batching samples from training set), disabling Masked Language Modeling (no BERT)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # No BERT
)

# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,  # This is all self-explanatory
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train model function
trainer.train()
print("Finshed transfer learning (phase 1 fine-tuning)...")

tracker.stop()
print("EmissionsTracker has completed process.")

# Evaluate the model on the test set after training
test_results = trainer.evaluate(eval_dataset=small_test_dataset)
print("Test results:", test_results)
