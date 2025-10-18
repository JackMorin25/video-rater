import os
import googleapiclient.discovery
from dotenv import load_dotenv

load_dotenv()

DEV_KEY = os.getenv("API_KEY")
api_service_name = "youtube"
api_version = "v3"

class ContentManager:

    def __init__(self):
        self.youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey = DEV_KEY
            )
        
    def getVidIdSearch(self, search):
        request = self.youtube.search().list(
            part="snippet",
            q=search,
            type="video",
            maxResults=25
        )

        response = request.execute()

        print(response)
    
    def getVidIdTrending(self):
        request = self.youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode="US",
            maxResults=25
        )

        response = request.execute()

        print(response)

    def listComments(self, vidID):
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=vidID,
            maxResults=20,
            order="relevance"
        )

        response = request.execute()
        
        for item in response['items']:
            print(item['snippet']['topLevelComment']['snippet']['textDisplay'])

        return response