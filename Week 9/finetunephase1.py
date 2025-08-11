import torch  # PyTorch for training purposes
import transformers
import datasets
from accelerate import Accelerator  # Need this to address numpy issues
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling)  # OpenAI and HuggingFace packages for training, training configs, and data batch preparation
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, interleave_datasets, disable_caching  # Hugging Face datasets
import json
import pandas as pd
import psutil
from codecarbon import EmissionsTracker

transformers.logging.set_verbosity_info()
disable_caching()
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
tokenizer.pad_token = tokenizer.eos_token # For Sequences
tokenizer.truncation = True # Truncating for Sequences used

#open_ds = load_dataset("OpenAssistant/oasst1",split='train',streaming=True).shuffle(seed=42).take(16_000)
open_ds = load_dataset("OpenAssistant/oasst1",split='train[:12000]')

#bbc_ds = load_dataset("permutans/fineweb-bbc-news","CC-MAIN-2022-33",split='train',streaming=True).shuffle(seed=42).take(20_000)
bbc_ds = load_dataset("permutans/fineweb-bbc-news","CC-MAIN-2022-33",split='train[:15000]')

#wiki_ds = load_dataset("NeuML/wikipedia-20250620",split="train",streaming=True).shuffle(seed=42).take(3_000)
wiki_ds = load_dataset("NeuML/wikipedia-20250620", split='train[:12000]') ## streaming=True)

# Could not get this to work.
#essen_ds = take_to_dataset(load_dataset("EssentialAI/essential-web-v1.0",split="train",  streaming=True), 3_000)
#essen_ds = load_dataset("EssentialAI/essential-web-v1.0",split='train',streaming=True).shuffle(seed=42).take(3_000)
#essen_ds = load_dataset("EssentialAI/essential-web-v1.0", split='train[:3000]') ## streaming=True)

#squad_ds = load_dataset("rajpurkar/squad_v2",split='train',streaming=True).shuffle(seed=42).take(15_000)
#squad_ds = load_dataset("rajpurkar/squad_v2",split='train[:11000]')
#squad_ds = load_dataset("rajpurkar/squad_v2", split='train[:8000]', trust_remote_code=True)

#eli_ds = load_dataset("sentence-transformers/eli5",split='train',streaming=True).shuffle(seed=42).take(20_000)
eli_ds = load_dataset("sentence-transformers/eli5",split='train[:8000]')

squad_raw = load_dataset("rajpurkar/squad_v2", split="train[:8000]")
dolly_raw = load_dataset("llm-wizard/dolly-15k-instruction-alpaca-format",split="train[:12000]")

#dolly_ds = load_dataset("llm-wizard/dolly-15k-instruction-alpaca-format",split='train',streaming=True).shuffle(seed=42).take(20_000)
def dolly_to_instruction(example):
    instr = example["instruction"] or ""
    inp   = example["input"] or ""
    out   = example["output"]
    if out is None or out.strip() == "":
        return None             # will be removed by filter
    prompt = f"{instr}\n{inp}".strip()
    return {"text": f"{prompt}\n\nAnswer:\n{out}"}

#dolly_raw = load_dataset("llm-wizard/dolly-15k-instruction-alpaca-format",split="train[:12000]")
dolly_ds = (dolly_raw.map(dolly_to_instruction, remove_columns=dolly_raw.column_names).filter(lambda ex: ex is not None))

def squad_to_instruction(example):
    answer_text = example["answers"]["text"][0] if example["answers"]["text"] and len(example["answers"]["text"]) > 0 else "I don't know."
    context = example["context"] if example["context"] is not None else ""
    question = example["question"] if example["question"] is not None else ""
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return {"text": f"{prompt} {answer_text}"}

squad_ds = squad_raw.map(squad_to_instruction,  remove_columns=squad_raw.column_names)
dolly_ds = (dolly_raw.map(dolly_to_instruction, remove_columns=dolly_raw.column_names)
                        .filter(lambda ex: ex is not None))
#p1datasets = interleave_datasets([open_ds, bbc_ds, squad_ds, eli_ds, dolly_ds, wiki_ds],seed=42, probabilities=None)

datasets_merge = [open_ds, bbc_ds, wiki_ds, squad_ds, eli_ds, dolly_ds]
p1datasets = concatenate_datasets([open_ds, bbc_ds, wiki_ds, squad_ds, eli_ds, dolly_ds])
eval_dataset  = p1datasets.select(range(1000))
train_dataset = p1datasets.select(range(1000, len(p1datasets)))

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
phase1 = p1datasets.map(preprocess_function, batched=True, remove_columns=p1datasets.column_names, keep_in_memory=True)
phase1.set_format("torch", ["input_ids","attention_mask"])

eval_strat  = phase1.select(range(1000))
train_strat = phase1.select(range(1000, len(phase1)))
# Evaluating Performance based on a split
#train_strat = train_strat.map(preprocess_function, remove_columns=train_strat.column_names)
#eval_strat  = eval_strat.map(preprocess_function,  remove_columns=eval_strat.column_names)

training_args = TrainingArguments(
    output_dir='./PhaseI-checkpoint',
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

# Trainer is set up to work with smaller datasets
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_strat,
    eval_dataset=eval_strat,
    processing_class=tokenizer,
)

# Train model function + EmissionsTracker.
#trainer.train()
trainer.train(resume_from_checkpoint="./PhaseI-checkpoint/checkpoint-1500")
print("Finshed transfer learning (phase 1 fine-tuning)...")
tracker.stop()
print("EmissionsTracker has completed process.")