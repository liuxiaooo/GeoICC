import json
import os
from PIL import Image
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import DefaultDataCollator
from PIL import ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset')

def generate_samples(path):
    """Generator function to yield image-label pairs from JSON data"""
    image_json = os.path.join(path, "image.json")
    with open(image_json, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        image_path = os.path.join(path, "image", key)
        image = Image.open(image_path)
        yield {'image': image, 'label': value}

def get_label_mappings(path):
    """Load label mappings from JSON file"""
    label_json = os.path.join(path, "label.json")
    with open(label_json, 'r') as f:
        data = json.load(f)
    label2id = {k: str(v) for k, v in data.items()}
    id2label = {str(v): k for k, v in data.items()}
    return label2id, id2label

# Create dataset and split into train/test
ds = Dataset.from_generator(generate_samples, gen_kwargs={"path": dataset_path})
ds = ds.train_test_split(test_size=0.2)
label2id, id2label = get_label_mappings(dataset_path)

# Initialize image processor and transforms
checkpoint = 'google/vit-base-patch16-224'
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def apply_transforms(examples):
    """Apply image transformations to dataset examples"""
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

ds = ds.with_transform(apply_transforms)

def compute_metrics(pred):
    """Calculate evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1, "recall": recall}

# Initialize model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=2619,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(script_dir, "models"),
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    greater_is_better=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

data_collator = DefaultDataCollator()

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()