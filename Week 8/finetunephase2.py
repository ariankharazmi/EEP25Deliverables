import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
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
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir) # Changed From GPT2LMHeadModel

# New Addition: Checking for Apple Silicon (M-Series) CPU if available: (EEP 2025 - Week 2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set padding token for consistency (End-of-Sequence token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = True # Truncating for Sequences used

# Load the fine-tuning dataset (Phase II) (Improving Curiosity-16's conversationality and Chain-of-Thought capabilities.)
## alpaca_ds = load_dataset('json', data_files='alpaca_data.json', split='train')
ar_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_ar", split='train')
lr_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_lr", split='train')
rc_ds = load_dataset("TAUR-Lab/Taur_CoT_Analysis_Project___gpt-4o-mini-2024-07-18", "agieval_lsat_rc", split='train')
openai_ds = load_dataset("openai/gsm8k", "main", split='train')
alpaca_ds = load_dataset("yahma/alpaca-cleaned", split='train[:17000]', trust_remote_code=True) ## streaming=True) ## Moved Alpaca Data to Phase I from Phase II

p2datasets = concatenate_datasets([ar_ds, lr_ds, rc_ds, openai_ds, alpaca_ds])
eval_dataset  = p2datasets.select(range(1000))
train_dataset = p2datasets.select(range(1000, len(p2datasets)))

def preprocess_function(batch):
    candidate_fields = ['text', 'question', 'prompt', 'response', 'chosen', 'rejected', 'content',
                   'sentence', 'concept_name', 'context', 'column', 'id', 'name',
                   'instruction', 'instances', 'input', 'noinput', 'output', 'inputid',
                   'answers', 'answer', 'title']
    texts = None
    for field in candidate_fields:
        if field in batch:
            col = batch[field]
            flat = [str(x) for x in col if x not in (None, "")]
            if flat:
                texts = flat
                break
    if texts is None:
        raise ValueError(f"No usable text in batch keys: {list(batch.keys())}")
    return tokenizer(texts,truncation=True,padding='max_length',max_length=256)
phase2 = p2datasets.map(preprocess_function, batched=True, remove_columns=p2datasets.column_names, keep_in_memory=True)
phase2.set_format("torch", ["input_ids","attention_mask"])

eval_strat  = phase2.select(range(1000))
train_strat = phase2.select(range(1000, len(phase2)))

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./PhaseII-checkpoint',
    eval_strategy='epoch', #Evaluation Strategy
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,  # Increase number of epochs (cycles of running through)
    #num_train_epochs=2,  # Increase number of epochs (cycles of running through)
    per_device_eval_batch_size=2,  # Smaller batch size for faster processing speeds/time
    weight_decay=0.01,
    save_total_limit=2,  # Number of checkpoints that will be saved to the filePATH
    #fp16=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,  # GPT-2 is not an encoding model.
)

# Trainer Function setup
# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_strat,
    eval_dataset=eval_strat,
    processing_class=tokenizer,
)

# Resume from checkpoint during training if needed to run fine-tuning in different intervals
#Add this snippet into train.train() if needed --> (resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16")
# Added to resume from Phase I checkpoint for Phase II
#trainer.train(resume_from_checkpoint="/Users/kharazmimac/PycharmProjects/Curiosity-16-run1/checkpoint")
trainer.train()

# Save the model
#trainer.save_model('./PhaseII-checkpoint')

print("Finshed transfer learning (phase 2 fine-tuning)...")
tracker.stop()
print("EmissionsTracker has completed process.")
# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)