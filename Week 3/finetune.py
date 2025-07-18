import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import transformers
transformers.logging.set_verbosity_info()
from codecarbon import EmissionsTracker


# Load the fine-tuning dataset (Alpaca data  for improving Curiosity-16's conversationality)
fine_tune_ds = load_dataset('json', data_files='alpaca_data.json', split='train')

tracker = EmissionsTracker()
tracker.start()
print("Emissions tracker has started.")

print("Beginning transfer learning (phase 2 fine-tuning)...")

# Load the pre-trained model and tokenizer from the checkpoint, training results @ /Users/kharazmimac/PycharmProjects/Curiosity-16
checkpoint_dir = '/Users/kharazmimac/PycharmProjects/Curiosity-16/checkpointetc'  # Adjust the path as necessary (Currently temporary)
model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)

# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set padding token for consistency
tokenizer.pad_token = tokenizer.eos_token

# Preprocess function for fine-tuning
def preprocess_function(dataset_column_examples):
    # Adjust this list based on your dataset columns
    text_fields = ['text', 'prompt', 'response', 'chosen', 'rejected', 'content',
        'sentence', 'concept_name', 'context',
        'column', 'id', 'name', 'instruction', 'instances',
        'input', 'noinput', 'output']
    for field in text_fields:
        if field in dataset_column_examples:
            texts = dataset_column_examples[field]
            break
    else:
        raise ValueError(f"No available text fields were found: {dataset_column_examples.keys()}")

    texts = [str(text) if text is not None else "" for text in texts]
    return tokenizer(texts, truncation=True, padding='max_length', max_length=256)

# Tokenize the fine-tuning dataset
tokenized_datasets = fine_tune_ds.map(preprocess_function, batched=True, remove_columns=fine_tune_ds.column_names)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask'])

dataset_size = len(tokenized_datasets)

# Define the size of the subsets, for training sets and eval sets, good for setting sizes later
eval_size = min(200, dataset_size)

# Shuffle and split the dataset
shuffled_dataset = tokenized_datasets.shuffle(seed=42)
small_eval_dataset = shuffled_dataset.select(range(eval_size))

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_results',
    num_train_epochs=3, 
    per_device_train_batch_size=2,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch', 
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500, 
)

# Data collator for language modeling (not using MLM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer setup
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
trainer.train(resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16")

# Save the model
trainer.save_model('./fine_tuned_model')

print("Finshed transfer learning (phase 2 fine-tuning)...")

tracker.stop()
print("EmissionsTracker has completed process.")

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)