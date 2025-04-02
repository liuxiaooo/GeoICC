from datasets import load_dataset, DatasetDict
from glob import glob
import random
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Set random seed for reproducibility
random.seed(42)

# Load and split dataset files
all_file_list = glob(pathname="yourcsv")  # Replace "yourcsv" with your actual file pattern
test_file_list = random.sample(all_file_list, 200)
train_file_list = [i for i in all_file_list if i not in test_file_list]

print(f"Train files: {len(train_file_list)}, Test files: {len(test_file_list)}")

# Load datasets from CSV files
raw_datasets = load_dataset(
    "csv",
    data_files={'train': train_file_list, 'valid': test_file_list},
    cache_dir="cache_data"
)

print("Raw datasets loaded:")
print(raw_datasets)

# Initialize tokenizer and set parameters
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# Test tokenizer on sample data
outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

# Add special tokens to tokenizer
tokenizer.add_special_tokens(special_tokens_dict={
    'bos_token': '<|endoftext|>',
    'eos_token': '<|endoftext|>',
    'unk_token': '<|endoftext|>'
})

# Define tokenization function
def tokenize(element):
    """Tokenize text content and return input_ids with proper length"""
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

# Apply tokenization to datasets
tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

print("Tokenized datasets:")
print(tokenized_datasets)

# Load or initialize GPT-2 model
GPT_MODEL_NAME_OR_PATH = "yuanzhoulvpi/gpt2_chinese"
GPT_model = GPT2LMHeadModel.from_pretrained(
    GPT_MODEL_NAME_OR_PATH,
    add_cross_attention=True
)

# Get model configuration
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize model
model = GPT_model
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# Set up data collator for language modeling
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Test data collator
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

# Configure training arguments
args = TrainingArguments(
    output_dir="gpt2_geo_test",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    eval_steps=4000,
    logging_steps=4000,
    logging_dir='logs_gpt2_geo_test',
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=2000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=4000,
    fp16=True,
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

# Start training
print("Starting training...")
trainer.train()
print("Training completed!")