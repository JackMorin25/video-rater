from content_manager import ContentManager
from sentiment_analyzer import SentimentManager
import keyboard
import time


content_manager = ContentManager()
sentiment_manager = SentimentManager()

while True:
    if keyboard.read_key() != 'f4':
        time.sleep(0.1)
        continue

    topic= input("please enter a videoid: ")
    response = content_manager.listComments(topic)
    for item in response['items']:
        print("msg: " + item['snippet']['topLevelComment']['snippet']['textDisplay'])
        sentiment_manager.test(item['snippet']['topLevelComment']['snippet']['textDisplay'])