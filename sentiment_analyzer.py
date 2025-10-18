from transformers import pipeline

class SentimentManager:

    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="google-bert/bert-base-uncased",
            truncation=True
            )
    
    def test(self):
        result = self.sentiment_pipeline("I HATE THIS!!!")
        print(result)