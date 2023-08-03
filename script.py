# Comment extraction
import pandas as pd
from api_key import API_KEY
from googleapiclient.discovery import build

my_api_key = API_KEY()

youtube = build("youtube", "v3", developerKey = my_api_key)

original_comments = []

def extract_comment(video_id, pages):
    if int(pages) > 2500:
        raise Exception("The number of pages to extract can't exceed 2500 due to quota limit.") 

    request = youtube.commentThreads().list(
        part = "snippet",
        videoId = video_id,
        maxResults = 100
    )
    response = request.execute()
    for item in response["items"]:
        original_comments.append(item["snippet"]["topLevelComment"]["snippet"]["textOriginal"])

    n = 1
    while n < int(pages):
        request = youtube.commentThreads().list(
            part = "snippet",
            videoId = video_id,
            maxResults = 100,
            pageToken = response["nextPageToken"]
        )
        response = request.execute()
        for item in response["items"]:
            original_comments.append(item["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
        n += 1

    return original_comments

extract_comment(input("Enter the id of video: "), input("Enter the number of pages to extract: "))

df = pd.DataFrame(original_comments, columns = ["Comments"])



# Comment translating
from googletrans import Translator

def translate_comment(comment):
    translator = Translator()
    comment_in_english = translator.translate(comment).text
    return comment_in_english

df["Comments"] = df["Comments"].apply(translate_comment)



# Comment processing
url_pattern = "https?://(www\.)?[-a-zA-Z0-9@:%._\\+~#?&/=]+\.[a-z]{2,6}[-a-zA-Z0-9()@:%._\\+~#?&/=]*"

df["Comments"] = (
    df["Comments"]
    .str.replace(url_pattern, "", regex = True)
    .str.replace("#\S+", "", regex = True)
    .str.replace("[^\w\s']+", "", regex = True)
    .str.replace("\d+", "", regex = True)
    .str.replace("_", " ")
    .str.replace("\s+", " ", regex = True)
    .str.strip()
    .str.lower()
)

from nltk.corpus import stopwords

stop_words = stopwords.words("english")

df["Comments"] = df["Comments"].apply(
    lambda comment: " ".join(word for word in comment.split() if word not in stop_words)
)

df["Comments"] = df[df["Comments"] != ""]
df = df.dropna()



# Word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_cloud = WordCloud(collocations = False, max_words = 30, background_color = "white")

word_cloud.generate(" ".join(df["Comments"]))

plt.imshow(word_cloud, interpolation = "bilinear")
plt.axis("off")
plt.show()



# Sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

df["Sentiment scores"] = df["Comments"].apply(lambda comment: analyzer.polarity_scores(comment))

df["Compound Scores"] = df["Sentiment scores"].apply(lambda sentiment_score: sentiment_score["compound"])

def identify_sentiment_of_comment(compound_score):
    if compound_score >= 0.2:
        return "positive"
    elif compound_score <= -0.2:
        return "negative"
    else:
        return "neutral"

df["Sentiments"] = df["Compound Scores"].apply(identify_sentiment_of_comment)

print("Average compound score:", df["Compound Scores"].mean())

sentiment_count_df = df["Sentiments"].value_counts().sort_index(ascending = False)

plt.bar(sentiment_count_df.index, sentiment_count_df.values, 0.4)
plt.title("Number of Comments by Sentiment")
plt.show()