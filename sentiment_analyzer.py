from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datasets import Dataset

class SentimentManager:

    def __init__(self):
        #!!!
        #ONLY RUN _train() IF YOu WANT TO TWEAK THE TRAINING OF THE MODEL OR ARE MISSING THE MODEL
        #THIS CAN TAKE A VERY LONG TIME TO RUN
        #self._train()
        #!!!

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="./sentiment-bert",
            tokenizer="bert-base-uncased",
            truncation=True
        )
    
    def test(self, text):
        result = self.sentiment_pipeline(text)
        print(result)
    
    def _train(self):
        df = pd.read_csv("./dataset/YoutubeCommentsDataSet.csv")
        df["label"] = df["Sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=3)

        def tokenize_function(examples):
            texts = [str(x) for x in examples["Comment"]]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
       
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total # of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        trainer.train()
        trainer.save_model("./sentiment-bert")