from transformers import (VisionEncoderDecoderModel,
                          ViTModel, GPT2LMHeadModel,
                          AutoTokenizer, ViTImageProcessor,
                          Trainer, TrainingArguments)
from typing import List, Any
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from datasets import load_dataset, Dataset
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers.integrations import TensorBoardCallback
from sklearn.model_selection import train_test_split
import codecs
import csv
import nltk
import jieba
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

# Define experiment names and model paths relative to the script location
exp_names = ['a_geovit_geogpt',
             'a_16384_geogpt',
             'a_32384_geogpt',
             'a_large16224_geogpt',
             'a_16224_geogpt',
             'a_geovit_geogpt_40',
             'a_geovit_geogpt_80',
             'a_geovit_gpt_wechsel',
             'a_geovit_gpt_chinese']

VIT_MODEL_NAME_OR_PATHs = ['../geovit/models/checkpoint-7500',
                           'google/vit-base-patch16-384',
                           'google/vit-base-patch32-384',
                           'google/vit-large-patch16-224',
                           'google/vit-base-patch16-224',
                           '../geovit/models/checkpoint-7500',
                           '../geovit/models/checkpoint-7500',
                           '../geovit/models/checkpoint-7500',
                           '../geovit/models/checkpoint-7500']

GPT_MODEL_NAME_OR_PATHs = ['../geogpt/gpt2_geo/checkpoint-128000',
                           '../geogpt/gpt2_geo/checkpoint-128000',
                           '../geogpt/gpt2_geo/checkpoint-128000',
                           '../geogpt/gpt2_geo/checkpoint-128000',
                           '../geogpt/gpt2_geo/checkpoint-128000',
                           '../geogpt/gpt2_geo_40/checkpoint-32000',
                           '../geogpt/gpt2_geo_80/checkpoint-84000',
                           'benjamin/gpt2-wechsel-chinese',
                           'yuanzhoulvpi/gpt2_chinese']

for exp_name, VIT_MODEL_NAME_OR_PATH, GPT_MODEL_NAME_OR_PATH in zip(exp_names, VIT_MODEL_NAME_OR_PATHs, GPT_MODEL_NAME_OR_PATHs):

    BLANK_FILE_PATH = '196blank.csv'
    picnum = 392
    BLANK_AVE_FILE_PATH = "10blank_grs.csv"
    zidian = 'glossary.csv'
    grs = 0
    csv_name = 'geo2k'
    os.mkdir(exp_name)

    data = pd.read_csv(os.path.join(exp_name, csv_name + '.csv'))

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Rename the split data files
    train_filename = csv_name + '_train.csv'
    test_filename = csv_name + '_test.csv'

    # Save the split data to new CSV files with BOM UTF-8 encoding
    with tqdm(total=2) as pbar:
        train_data.to_csv(codecs.open(os.path.join(exp_name, train_filename), 'w', 'utf-8-sig'), index=False)
        pbar.update(1)
        test_data.to_csv(codecs.open(os.path.join(exp_name, test_filename), 'w', 'utf-8-sig'), index=False)
        pbar.update(1)

    VIT_model = ViTModel.from_pretrained(VIT_MODEL_NAME_OR_PATH)
    GPT_model = GPT2LMHeadModel.from_pretrained(GPT_MODEL_NAME_OR_PATH, add_cross_attention=True)
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME_OR_PATH)
    if exp_name == 'a_geovit_gpt_wechsel':
        tokenizer.pad_token = tokenizer.eos_token

    def process_image_2_pixel_value(x: str) -> Tensor:
        image = Image.open(x)
        res = processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return res

    def process_text_2_input_id(x: str):
        res = tokenizer(text=x, max_length=100, truncation=True, padding="max_length")['input_ids']
        return res

    new_encoder_decoder_model = VisionEncoderDecoderModel(
        encoder=VIT_model,
        decoder=GPT_model,
    )
    new_encoder_decoder_model.config.decoder_start_token_id = tokenizer.bos_token_id
    new_encoder_decoder_model.config.pad_token_id = tokenizer.pad_token_id
    new_encoder_decoder_model.config.add_cross_attention = True

    dataset = Dataset.from_pandas(df=pd.read_csv(os.path.join(exp_name, train_filename)))
    dataset = dataset.train_test_split(test_size=0.25)

    def tokenizer_text(examples):
        examples['labels'] = [process_text_2_input_id(i) for i in examples['text']]
        return examples

    def transform_images(examples):
        images = [process_image_2_pixel_value(i) for i in examples['image_path']]
        examples['pixel_values'] = images
        return examples

    dataset = dataset.map(
        function=tokenizer_text,
        batched=True
    )
    dataset.set_transform(transform=transform_images)

    def collate_fn(examples):
        pixel_values = torch.stack([i['pixel_values'] for i in examples])
        labels = torch.tensor([example["labels"] for example in examples], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    train_argument = TrainingArguments(
        output_dir=exp_name,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_strategy='epoch',
        logging_dir='logs_' + exp_name,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
        save_strategy="epoch",  # Save models by epoch
        fp16=True,
        remove_unused_columns=False,
        save_total_limit=10,
    )

    trainer = Trainer(
        model=new_encoder_decoder_model,
        args=train_argument,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collate_fn,
        callbacks=[TensorBoardCallback()]
    )
    trainer.train()

    print(exp_name)
    print("Training completed, starting evaluation")

    models_root = os.path.join(exp_name, 'models')

    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME_OR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(GPT_MODEL_NAME_OR_PATH)

    e = 0

    contents = os.listdir(models_root)
    folders = [item for item in contents if os.path.isdir(os.path.join(models_root, item))]
    folders.sort(key=lambda x: int(x[11:]))

    with open(BLANK_AVE_FILE_PATH, "r", encoding="utf-8") as avefile:
        avecsv_reader = csv.reader(avefile)
        averows = list(avecsv_reader)

        for model_name in folders:
            e += 1
            vision_encoder_decoder_model_name_or_path = os.path.join(models_root, model_name)
            model = VisionEncoderDecoderModel.from_pretrained(vision_encoder_decoder_model_name_or_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            max_length = 20
            num_beams = 4
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

            def predict(image_paths):
                images = []
                for image_path in image_paths:
                    i_image = Image.open(image_path)
                    if i_image.mode != "RGB":
                        i_image = i_image.convert(mode="RGB")
                    images.append(i_image)

                pixel_values = processor(images=images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)

                output_ids = model.generate(pixel_values, **gen_kwargs)

                preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                preds = [pred.strip() for pred in preds]
                return preds

            with open(BLANK_FILE_PATH, "r", encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

            sumbleu1 = 0
            sumbleu2 = 0
            sumbleu2s = 0
            sumbleu3 = 0
            sumbleu3s = 0
            sumbleu4 = 0
            sumbleu4s = 0
            sumMETEOR = 0
            sumROUGE1r = 0
            sumROUGE1p = 0
            sumROUGE1f = 0
            sumROUGE2r = 0
            sumROUGE2p = 0
            sumROUGE2f = 0
            sumROUGELr = 0
            sumROUGELp = 0
            sumROUGELf = 0
            sumjiebableu1 = 0
            sumjiebableu2 = 0
            sumjiebableu2s = 0
            sumjiebableu3 = 0
            sumjiebableu3s = 0
            sumjiebableu4 = 0
            sumjiebableu4s = 0
            sumjiebaMETEOR = 0
            sumjiebaROUGE1r = 0
            sumjiebaROUGE1p = 0
            sumjiebaROUGE1f = 0
            sumjiebaROUGE2r = 0
            sumjiebaROUGE2p = 0
            sumjiebaROUGE2f = 0
            sumjiebaROUGELr = 0
            sumjiebaROUGELp = 0
            sumjiebaROUGELf = 0
            sumgrs = 0

            for i, row in tqdm(enumerate(rows), total=len(rows) - 1):
                if i == 0:
                    continue
                image_path = row[0]
                reference = row[1]
                infer = predict([image_path])[0]
                infer = infer.replace(' ', '')
                rows[i][2] = infer

                # Calculate BLEU scores
                reference_tokens_jieba = jieba.lcut(reference)
                rows[i][3] = reference_tokens_jieba
                infer_tokens_jieba = jieba.lcut(infer)
                rows[i][4] = infer_tokens_jieba

                reference_tokens = list(reference)
                rows[i][5] = reference_tokens
                infer_tokens = list