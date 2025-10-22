from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

class SentimentManager:

    def __init__(self):
        self._train()

        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="./sentiment-bert",
            tokenizer="bert-base-uncased",
            truncation=True
        )
    
    def test(self):
        result = self.sentiment_pipeline("I HATE THIS!!!")
        print(result)
    
    def _train(self):
        model = BertForSequenceClassification.from_pretrained("bert-large-uncased")
        df = pd.read_csv("./dataset/YoutubeCommentsDataSet.csv")
        comments = df['Comment']
        labels = df['Sentiment']
        train_texts, test_texts, train_labels, test_labels = train_test_split(comments, labels, test_size=0.2, random_state=42)

        

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
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=df,         # training dataset
            eval_dataset=df            # evaluation dataset
        )

        trainer.train()
        trainer.save_model("./sentiment-bert")