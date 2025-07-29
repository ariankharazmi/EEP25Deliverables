import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
import datasets, random
from datasets import load_dataset, concatenate_datasets
import transformers


transformers.logging.set_verbosity_info()
from codecarbon import EmissionsTracker

# Emissions Tracker for tracking Emissions of fine-tune Phase II process.
tracker = EmissionsTracker()
tracker.start()
print("Emissions tracker has started.")

# Load the pre-trained model and tokenizer from the checkpoint, training results @ /Users/ariankharazmi/PycharmProjects/Curiosity-16
checkpoint_dir = '/Users/ariankharazmi/PycharmProjects/Curiosity-16-run1/checkpoint'  # Adjust the path as necessary (Currently temporary)
model_name = 'gpt2-medium' # GPT-2 Medium instead of GPT-2Small (difference from previous Curiosity models. More parameters.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir) # Changed From GPT2LMHeadModel

# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set padding token for consistency (End-of-Sequence token)
tokenizer.pad_token = tokenizer.eos_token

# Load the fine-tuning dataset (Phase II) (Improving Curiosity-16's conversationality and Chain-of-Thought capabilities.)
## alpaca_ds = load_dataset('json', data_files='alpaca_data.json', split='train')
ar_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_ar", split='train')
lr_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_lr", split='train')
rc_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_rc", split='train')
openai_ds = load_dataset("openai/gsm8k", "main", split='train')

combined_datasets = concatenate_datasets([])

# Preprocess function for fine-tuning
def preprocess_function(dataset_column_examples):
    # Adjust this list based on your dataset columns
    text_fields = ['text', 'prompt', 'response', 'chosen', 'rejected', 'content',
        'sentence', 'concept_name', 'context',
        'column', 'id', 'name', 'instruction', 'instances',
        'input', 'noinput', 'output', 'positive', 'negative']
    for field in text_fields:
        if field in dataset_column_examples:
            texts = dataset_column_examples[field]
            break
    else:
        raise ValueError(f"No available text fields were found in one or more datasets.: {dataset_column_examples.keys()}")
    texts = [str(text) if text is not None else "" for text in texts]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=256)

tokenized_datasets = combined_datasets.map(preprocess_function, batched=True, remove_columns=combined_datasets.column_names)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])
dataset_size = len(tokenized_datasets)
# Define the size of the subsets, for training sets and eval sets, good for setting sizes later
eval_size = min(400, dataset_size)

# Shuffle and split the dataset
shuffled_dataset = tokenized_datasets.shuffle(seed=42)
small_eval_dataset = shuffled_dataset.select(range(eval_size))

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_results',
    fp16=True,
    num_train_epochs=3,
    per_device_train_batch_size=4, # Changed from two (last year's Curiosity-14) to four batches (for Curiosity-16)
    gradient_accumulation_steps=2,
    save_total_limit=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    logging_dir='./logs',
    logging_steps=25,
    save_steps=500,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Trainer Function setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    eval_dataset=small_eval_dataset
)

# Resume from checkpoint during training if needed to run fine-tuning in different intervals
#Add this snippet into train.train() if needed --> (resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16")
trainer.train(resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16-run1/checkpoint")

# Save the model
trainer.save_model('./fine_tuned_model')

print("Finshed transfer learning (phase 2 fine-tuning)...")

tracker.stop()
print("EmissionsTracker has completed process.")

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)