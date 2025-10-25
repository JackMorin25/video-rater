from content_manager import ContentManager
from sentiment_analyzer import SentimentManager


content_manager = ContentManager()
sentiment_manager = SentimentManager()

#content_manager.getVidIdSearch("Ludwig")
response = content_manager.listComments("jBx4qIGd7BE")

for item in response['items']:
    print("msg: " + item['snippet']['topLevelComment']['snippet']['textDisplay'])
    sentiment_manager.test(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    
#sentiment_manager.test()