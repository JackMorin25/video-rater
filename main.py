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

    topic= input("please search a topic: ")
    response = content_manager.getVidIdSearch(topic)
    for item in response['items']:
        print("msg: " + item['snippet']['topLevelComment']['snippet']['textDisplay'])
        sentiment_manager.test(item['snippet']['topLevelComment']['snippet']['textDisplay'])