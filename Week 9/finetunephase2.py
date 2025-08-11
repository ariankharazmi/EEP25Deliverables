import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback)
import datasets, random
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, interleave_datasets, disable_caching
import transformers
transformers.logging.set_verbosity_info()
from codecarbon import EmissionsTracker
import psutil

# Emissions Tracker for tracking Emissions of fine-tune Phase II process.
transformers.logging.set_verbosity_info()
disable_caching()
tracker = EmissionsTracker()
tracker.start()
print("Emissions tracker has started.")

# Load the pre-trained model and tokenizer from the checkpoint, training results @ /Users/ariankharazmi/PycharmProjects/Curiosity-16
checkpoint_dir = '/Users/ariankharazmi/PycharmProjects/Curiosity16-run1/PhaseI-checkpoint/checkpoint-12375'  # Adjust the path as necessary (Currently temporary)
transformer_model_name = 'gpt2-medium' # GPT-2 Medium instead of GPT-2Small (difference from previous Curiosity models. More parameters.
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
# model = AutoModelForCausalLM.from_pretrained(checkpoint_dir) # Changed From GPT2LMHeadModel

# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to(device)
model.gradient_checkpointing_enable()

# Set padding token for consistency (End-of-Sequence token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True # Truncating for Sequences used

# Load the fine-tuning dataset (Phase II) (Improving Curiosity-16's conversationality and Chain-of-Thought capabilities.)
## alpaca_ds = load_dataset('json', data_files='alpaca_data.json', split='train')
ar_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_ar", split='latest[:8000]')
lr_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_lr", split='latest[:8000]')
rc_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_rc", split='latest[:8000]')
openai_ds = load_dataset("openai/gsm8k", "main", split='train[:10000]')
alpaca_ds = load_dataset("yahma/alpaca-cleaned", split='train[:52000]')
alpaca_ds = alpaca_ds.rename_column("output", "text")

p2datasets = concatenate_datasets([ar_ds, lr_ds, rc_ds, openai_ds, alpaca_ds]).shuffle(seed=42)
eval_dataset  = p2datasets.select(range(800))
train_dataset = p2datasets.select(range(1000, len(p2datasets)))

system_prompt = "You are a helpful AI assistant. If you are not certain, say 'I'm not sure.'"
def preprocess_function(batch):
    user_keys = ['text', 'question', 'prompt',
                    'sentence', 'concept_name', 'context', 'column', 'id', 'name',
                    'instruction', 'instances', 'input', 'noinput', 'output', 'inputid',
                    'answers', 'answer', 'title']
    assistant_keys = ['text', 'response', 'chosen', 'content', 'context',
                      'sentence', 'concept_name', 'context', 'column', 'id', 'name',
                      'instruction', 'instances', 'input',
                      'noinput', 'output', 'inputid',
                      'answers', 'answer', 'title']
    texts = []
    for field in user_keys:
        if field in batch:
            texts = [system_prompt + x.strip() + tokenizer.eos_token
                     for x in batch[field]
                     if x and x.strip() and len(x.strip()) > 10]
            if texts:
                break
    for field in assistant_keys:
        if field in batch:
            texts = [system_prompt + x.strip() + tokenizer.eos_token
                     for x in batch[field]
                     if x and x.strip() and len(x.strip()) > 10]
            if texts:
                break
    if not texts:
        raise ValueError(f"No usable text in batch keys: {batch.keys()}")
    tokenized = tokenizer(texts, truncation=True, padding='longest', add_special_tokens=True, max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
phase2 = p2datasets.map(preprocess_function, batched=True, remove_columns=p2datasets.column_names)
phase2.set_format("torch", ["input_ids", "attention_mask", "labels"])
#keep_in_memory=True)
phase2.set_format("torch", ["input_ids","attention_mask"])

eval_strat  = phase2.select(range(800))
train_strat = phase2.select(range(1000, len(phase2)))

early_stop = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01,
)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./PhaseII-checkpoint',
    lr_scheduler_type="linear",
    eval_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    logging_steps=500,
    metric_for_best_model='eval_loss',
    logging_first_step=True,
    load_best_model_at_end=True,
    num_train_epochs=2,  # Increase number of epochs (cycles of running through)
    #num_train_epochs=2,  # Increase number of epochs (cycles of running through)
    per_device_eval_batch_size=4,  # Smaller batch size for faster processing speeds/time
    weight_decay=0.01,
    save_total_limit=3,  # Number of checkpoints that will be saved to the filePATH
    greater_is_better=False,
    #ignore_data_skip = True
#fp16=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,  # GPT-2 is not an encoding model.
)

# Trainer Function setup
# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[early_stop],
    data_collator=data_collator,
    train_dataset=train_strat,
    eval_dataset=eval_strat,
    processing_class=tokenizer,
)

# Resume from checkpoint during training if needed to run fine-tuning in different intervals
#Add this snippet into train.train() if needed --> (resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16")
# Added to resume from Phase I checkpoint for Phase II
#trainer.train(resume_from_checkpoint="/Users/ariankharazmi/PycharmProjects/Curiosity16-run1/PhaseII-checkpoint/checkpoint-18720")
trainer.train()

# Save the model
#trainer.save_model('./PhaseII-checkpoint')

print("Finshed transfer learning (phase 2 fine-tuning)...")
tracker.stop()
print("EmissionsTracker has completed process.")
# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)