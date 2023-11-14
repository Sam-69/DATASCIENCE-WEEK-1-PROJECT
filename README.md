# DATASCIENCE-WEEK-1-PROJECT
Question 2). Let’s say you’re a Product Data Scientist at Instagram. How would you measure the success of the Instagram TV product?

In my role as a Product Data Scientist at Instagram, I'd have a bunch of ways to gauge how well IGTV is doing. Think of it as checking the pulse of this feature. We're talking about metrics, which are basically numbers that tell us what's going on. Here's what I'd look at:

(1)Views: This one's simple. It's about how many folks watched at least a bit of an IGTV video. We'd want to know if more people are tuning in over time.

(2)Average Watch Time: This is the average amount of time people spend watching IGTV videos. We'd want to see if folks are sticking around longer.

(3)Audience Retention: This tells us how many people watched an entire IGTV video from start to finish. Ideally, we'd like to see this number go up.

(4)Engagement Rate: We're talking about likes, comments, and shares here. It's important to know how often people are interacting with IGTV videos.

(5)Follower Growth: This is about how many new followers IGTV creators are getting. It's a good sign if this number keeps going up.

(Revenue: Money talks, right? We'd keep tabs on how much cash Instagram is making from IGTV ads.

Now, beyond these core metrics, there are some other things I'd keep an eye on:

(1)Top IGTV Creators: Who's getting the most views, followers, and engagement? That can tell us what kind of content is really hitting the mark.

(2)Popular IGTV Categories: What types of IGTV videos are people loving the most? We'd want to know what categories are trending.

(3)IGTV Discovery Features: How effective are the features that help folks find IGTV content? We'd study stuff like the Explore feed and recommendations.

(4)IGTV User Satisfaction: Are people happy with IGTV? This matters a lot, and we'd want to know if folks are enjoying the product.

So, what's the point of all this data tracking? Well, it helps us figure out what's working and what's not with IGTV. For instance:

(1)If we see views going up, we know people are tuning in more.
(2)If the average watch time is increasing, it means people are finding content they like.
(3)If engagement rates are high, it means users are getting involved.


Jupyter Notebook
user-behavior-on-instagram

Code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[1], line 5
      1 # This Python 3 environment comes with many helpful analytics libraries installed
      2 # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
      3 # For example, here's several helpful packages to load
----> 5 import numpy as np # linear algebra
      6 import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
      8 # Input data files are available in the read-only "../input/" directory
      9 # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

ModuleNotFoundError: No module named 'numpy'

# Preliminary data exploration
​
# 1. Check the overall size of the dataset
num_comments = df_comments.shape[0]
​
# 2. Look at the distribution of comments per user and per photo
comments_per_user = df_comments['User  id'].value_counts()
comments_per_photo = df_comments['Photo id'].value_counts()
​
# 3. Examine the distribution of the `Hashtags used count` column
hashtags_dist = df_comments['Hashtags used count'].value_counts()
​
# 4. Check the frequency of emoji usage in comments
emoji_usage = df_comments['emoji used'].value_counts()
​
# 5. Check the date range of the comments
df_comments['created Timestamp'] = pd.to_datetime(df_comments['created Timestamp'])
date_range = df_comments['created Timestamp'].min(), df_comments['created Timestamp'].max()
​
num_comments, comments_per_user.describe(), comments_per_photo.describe(), hashtags_dist, emoji_usage, date_range
​
import pandas as pd
​
# Define file path
file_path = os.path.join('/kaggle/input/user-behavior-on-instagram/comments_cleaned.csv')
​
# Load the data into a pandas DataFrame
df_comments = pd.read_csv(file_path)
​
# Display the first few rows of the DataFrame
df_comments.head()
​
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[2], line 1
----> 1 import pandas as pd
      3 # Define file path
      4 file_path = os.path.join('/kaggle/input/user-behavior-on-instagram/comments_cleaned.csv')

ModuleNotFoundError: No module named 'pandas'

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
​
# Concatenate all comments into a single string
all_comments = " ".join(comment for comment in df_comments.comment)
​
# Create a word cloud
wordcloud = WordCloud(background_color="white").generate(all_comments)
​
# Display the word cloud
plt.figure(figsize=(8,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
​
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[3], line 1
----> 1 from wordcloud import WordCloud
      2 import matplotlib.pyplot as plt
      3 from PIL import Image

ModuleNotFoundError: No module named 'wordcloud'

from textblob import TextBlob
​
# Apply TextBlob to each comment to get sentiment polarity
df_comments['sentiment_polarity'] = df_comments['comment'].apply(lambda text: TextBlob(text).sentiment.polarity)
​
# Classify sentiment as positive, neutral, or negative based on polarity
df_comments['sentiment'] = df_comments['sentiment_polarity'].apply(lambda p: 'positive' if p > 0 else ('negative' if p < 0 else 'neutral'))
​
# Display the first few rows of the DataFrame
df_comments.head()
​
from nltk.sentiment import SentimentIntensityAnalyzer
​
# Initialize the sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()
​
# Apply the analyzer to each comment to get sentiment scores
df_comments['sentiment_scores'] = df_comments['comment'].apply(sia.polarity_scores)
​
# Extract compound scores to a separate column
df_comments['compound_score'] = df_comments['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
​
# Classify sentiment as positive, neutral, or negative based on compound score
df_comments['sentiment'] = df_comments['compound_score'].apply(lambda c: 'positive' if c > 0.05 else ('negative' if c < -0.05 else 'neutral'))
​
# Display the first few rows of the DataFrame
df_comments.head()
​
# Let's see the overall sentiment distribution
sentiment_distribution = df_comments['sentiment'].value_counts()
sentiment_distribution
​
# Fetch negative comments
negative_comments = df_comments[df_comments['sentiment'] == 'negative']
​
# Display a few negative comments
negative_comments_sample = negative_comments.sample(10)
​
negative_comments_sample
​
# Concatenate all negative comments into a single string
all_negative_comments = " ".join(comment for comment in negative_comments.comment)
​
# Create a word cloud
wordcloud_negative = WordCloud(background_color="white").generate(all_negative_comments)
​
# Display the word cloud
plt.figure(figsize=(8,6))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.show()
​
# Calculate the proportions of positive, negative, and neutral comments that contain emojis
emoji_sentiment_proportions = df_comments.groupby('sentiment')['emoji used'].value_counts(normalize=True).unstack()
​
# Convert to percentages
emoji_sentiment_proportions = emoji_sentiment_proportions * 100
​
emoji_sentiment_proportions
​
# Visualize the proportions
emoji_sentiment_proportions.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Proportion of Comments With and Without Emojis by Sentiment')
plt.ylabel('Percentage')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()
​
# Count the number of comments per user
user_comment_counts = df_comments['User  id'].value_counts()
​
# Identify the 10 most active users
top_users = user_comment_counts.nlargest(10)
​
# For each of the top users, calculate the sentiment distribution of their comments
top_user_sentiments = df_comments[df_comments['User  id'].isin(top_users.index)].groupby('User  id')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
​
# Convert to percentages
top_user_sentiments = top_user_sentiments * 100
​
top_users, top_user_sentiments
​
# Count the number of comments per photo
photo_comment_counts = df_comments['Photo id'].value_counts()
​
# Identify the 10 photos that received the most comments
top_photos = photo_comment_counts.nlargest(10)
​
# For each of the top photos, calculate the sentiment distribution of their comments
top_photo_sentiments = df_comments[df_comments['Photo id'].isin(top_photos.index)].groupby('Photo id')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
​
# Convert to percentages
top_photo_sentiments = top_photo_sentiments * 100
​
top_photos, top_photo_sentiments
​
# Calculate the average sentiment polarity for comments with different numbers of hashtags
hashtag_sentiment = df_comments.groupby('Hashtags used count')['sentiment_polarity'].mean()
​
# Visualize the relationship
hashtag_sentiment.plot(kind='bar', figsize=(10, 6))
plt.title('Average Sentiment Polarity by Number of Hashtags Used')
plt.ylabel('Average Sentiment Polarity')
plt.xlabel('Number of Hashtags Used')
plt.xticks(rotation=0)
plt.show()
​
# Calculate the length of each comment in terms of the number of words and characters
df_comments['word_count'] = df_comments['comment'].apply(lambda text: len(text.split()))
df_comments['char_count'] = df_comments['comment'].apply(len)
​
# Calculate the average length of comments for each sentiment category
average_lengths = df_comments.groupby('sentiment')[['word_count', 'char_count']].mean()
​
average_lengths
​
# Visualize the average number of words in comments for each sentiment category
average_lengths['word_count'].plot(kind='bar', figsize=(8, 6))
plt.title('Average Number of Words in Comments by Sentiment')
plt.ylabel('Average Number of Words')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()
​
# Visualize the average number of characters in comments for each sentiment category
average_lengths['char_count'].plot(kind='bar', figsize=(8, 6))
plt.title('Average Number of Characters in Comments by Sentiment')
plt.ylabel('Average Number of Characters')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()
​
# Calculate the average length of comments that use emojis and those that don't
average_lengths_emoji = df_comments.groupby('emoji used')[['word_count', 'char_count']].mean()
​
average_lengths_emoji
​
# Visualize the average number of words in comments that use emojis and those that don't
average_lengths_emoji['word_count'].plot(kind='bar', figsize=(8, 6))
plt.title('Average Number of Words in Comments by Emoji Use')
plt.ylabel('Average Number of Words')
plt.xlabel('Emoji Used')
plt.xticks(rotation=0)
plt.show()
​
# Visualize the average number of characters in comments that use emojis and those that don't
average_lengths_emoji['char_count'].plot(kind='bar', figsize=(8, 6))
plt.title('Average Number of Characters in Comments by Emoji Use')
plt.ylabel('Average Number of Characters')
plt.xlabel('Emoji Used')
plt.xticks(rotation=0)
plt.show()
​
# Calculate the distribution of the number of hashtags used in the comments
hashtag_usage_counts = df_comments['Hashtags used count'].value_counts()
​
# Sort by the number of hashtags
hashtag_usage_counts = hashtag_usage_counts.sort_index()
​
hashtag_usage_counts
​
# Visualize the distribution of the number of hashtags used in the comments
hashtag_usage_counts.plot(kind='bar', figsize=(8, 6))
plt.title('Distribution of the Number of Hashtags Used in Comments')
plt.ylabel('Number of Comments')
plt.xlabel('Number of Hashtags Used')
plt.xticks(rotation=0)
plt.show()
​
# For each user, calculate the distribution of their comments by sentiment
user_sentiment_distribution = df_comments.groupby('User  id')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
​
# Convert to percentages
user_sentiment_distribution = user_sentiment_distribution * 100
​
# Identify the users who have the highest proportions of negative and neutral comments
user_with_most_negative_comments = user_sentiment_distribution['negative'].idxmax()
user_with_most_neutral_comments = user_sentiment_distribution['neutral'].idxmax()
​
user_with_most_negative_comments, user_with_most_neutral_comments, user_sentiment_distribution.loc[[user_with_most_negative_comments, user_with_most_neutral_comments]]
​
# For each user, calculate the distribution of their comments by sentiment
user_sentiment_distribution = df_comments.groupby('User  id')['sentiment'].value_counts(normalize=True).unstack().fillna(0)
​
# Convert to percentages
user_sentiment_distribution = user_sentiment_distribution * 100
​
# Identify the users who have the highest proportions of negative and neutral comments
user_with_most_negative_comments = user_sentiment_distribution['negative'].idxmax()
user_with_most_neutral_comments = user_sentiment_distribution['neutral'].idxmax()
​
user_with_most_negative_comments, user_with_most_neutral_comments, user_sentiment_distribution.loc[[user_with_most_negative_comments, user_with_most_neutral_comments]]
​
# Recalculate necessary variables for the final visualization
sentiment_counts = df_comments['sentiment'].value_counts()
emoji_sentiment_proportions = df_comments.groupby('sentiment')['emoji used'].value_counts(normalize=True).unstack().fillna(0) * 100
​
# Create the final visualization
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
​
# Distribution of comments by sentiment
sentiment_counts.plot(kind='bar', ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title('Distribution of Comments by Sentiment')
axs[0, 0].set_xlabel('Sentiment')
axs[0, 0].set_ylabel('Number of Comments')
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=0)
​
# Proportion of comments with emojis by sentiment
emoji_sentiment_proportions.plot(kind='bar', stacked=True, ax=axs[0, 1])
axs[0, 1].set_title('Proportion of Comments With Emojis by Sentiment')
axs[0, 1].set_xlabel('Sentiment')
axs[0, 1].set_ylabel('Percentage')
axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=0)
​
# Distribution of comments by the number of hashtags used
hashtag_usage_counts.plot(kind='bar', ax=axs[1, 0], color='skyblue')
axs[1, 0].set_title('Distribution of Comments by Number of Hashtags Used')
axs[1, 0].set_xlabel('Number of Hashtags Used')
axs[1, 0].set_ylabel('Number of Comments')
axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=0)
​
# Average number of characters in comments by sentiment
average_lengths['char_count'].plot(kind='bar', ax=axs[1, 1], color='skyblue')
axs[1, 1].set_title('Average Number of Characters in Comments by Sentiment')
axs[1, 1].set_xlabel('Sentiment')
axs[1, 1].set_ylabel('Average Number of Characters')
axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=0)
​
plt.tight_layout()
plt.show()
​
# For each user, calculate the total number of comments and the distribution of their comments by sentiment
user_comments_sentiment = df_comments.groupby('User  id')['sentiment'].value_counts().unstack().fillna(0)
​
# Calculate the total number of comments for each user
user_comments_sentiment['total_comments'] = user_comments_sentiment.sum(axis=1)
​
# Sort by the total number of comments
user_comments_sentiment = user_comments_sentiment.sort_values('total_comments', ascending=False)
​
# Identify the users who commented the most
top_commenting_users = user_comments_sentiment.head(10)
​
# Identify the users who commented the least
least_commenting_users = user_comments_sentiment.tail(10)
​
top_commenting_users, least_commenting_users
​
# Create pie charts showing the sentiment distribution for the most frequent and least frequent commenters
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
​
# Most frequent commenters
top_commenting_users[['negative', 'neutral']].mean().plot(kind='pie', ax=axs[0], autopct='%1.1f%%')
axs[0].set_ylabel('')
axs[0].set_title('Sentiment Distribution for the Most Frequent Commenters')
​
# Least frequent commenters
least_commenting_users[['negative', 'neutral']].mean().plot(kind='pie', ax=axs[1], autopct='%1.1f%%')
axs[1].set_ylabel('')
axs[1].set_title('Sentiment Distribution for the Least Frequent Commenters')
​
plt.show()
​
# For each user, calculate the average number of hashtags used in their comments
average_hashtags_by_user = df_comments.groupby('User  id')['Hashtags used count'].mean()
​
# Identify the users who use the most and the least number of hashtags on average
user_with_most_hashtags = average_hashtags_by_user.idxmax()
user_with_least_hashtags = average_hashtags_by_user.idxmin()
​
user_with_most_hashtags, user_with_least_hashtags, average_hashtags_by_user.loc[[user_with_most_hashtags, user_with_least_hashtags]]
​
# For each user, for each number of hashtags used, calculate the distribution of their comments by sentiment
user_hashtag_sentiment_distribution = df_comments.groupby(['User  id', 'Hashtags used count'])['sentiment'].value_counts(normalize=True).unstack().fillna(0)
​
# Convert to percentages
user_hashtag_sentiment_distribution = user_hashtag_sentiment_distribution * 100
​
# Identify the users and the number of hashtags used that have the highest proportions of negative and neutral comments
user_hashtag_with_most_negative_comments = user_hashtag_sentiment_distribution['negative'].idxmax()
user_hashtag_with_most_neutral_comments = user_hashtag_sentiment_distribution['neutral'].idxmax()
​
user_hashtag_with_most_negative_comments, user_hashtag_with_most_neutral_comments, user_hashtag_sentiment_distribution.loc[[user_hashtag_with_most_negative_comments, user_hashtag_with_most_neutral_comments]]
​
# Visualize the sentiment distribution for the user and number of hashtags with the highest proportion of negative comments
user_hashtag_sentiment_distribution.loc[[user_hashtag_with_most_negative_comments]].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title(f'Sentiment Distribution for User {user_hashtag_with_most_negative_comments[0]} Using {user_hashtag_with_most_negative_comments[1]} Hashtags')
plt.ylabel('Percentage')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()
​
# Visualize the sentiment distribution for the user and number of hashtags with the highest proportion of neutral comments
user_hashtag_sentiment_distribution.loc[[user_hashtag_with_most_neutral_comments]].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title(f'Sentiment Distribution for User {user_hashtag_with_most_neutral_comments[0]} Using {user_hashtag_with_most_neutral_comments[1]} Hashtags')
plt.ylabel('Percentage')
plt.xlabel('Sentiment')
plt.xticks(rotation=0)
plt.show()
​
# For each user, for comments with and without emojis, calculate the average number of hashtags used
user_emoji_hashtag_usage = df_comments.groupby(['User  id', 'emoji used'])['Hashtags used count'].mean()
​
# Identify the users and the emoji usage (yes or no) that have the highest and lowest average number of hashtags used
user_emoji_with_most_hashtags = user_emoji_hashtag_usage.idxmax()
user_emoji_with_least_hashtags = user_emoji_hashtag_usage.idxmin()
​
user_emoji_with_most_hashtags, user_emoji_with_least_hashtags, user_emoji_hashtag_usage.loc[[user_emoji_with_most_hashtags, user_emoji_with_least_hashtags]]
​
# Calculate the average number of characters in comments for each number of hashtags used
average_length_by_hashtag_count = df_comments.groupby('Hashtags used count')['char_count'].mean()
​
# Visualize this relationship
average_length_by_hashtag_count.plot(kind='bar', figsize=(8, 6), color='skyblue')
plt.title('Average Number of Characters in Comments for Each Number of Hashtags Used')
plt.ylabel('Average Number of Characters')
plt.xlabel('Number of Hashtags Used')
plt.xticks(rotation=0)
plt.show()
​
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
​
# Average Number of Characters in Comments for Each Number of Hashtags Used
average_length_by_hashtag_count.plot(kind='bar', ax=axs[0, 0], color='skyblue')
axs[0, 0].set_title('Average Number of Characters in Comments for Each Number of Hashtags Used')
axs[0, 0].set_xlabel('Number of Hashtags Used')
axs[0, 0].set_ylabel('Average Number of Characters')
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=0)
​
# Sentiment Distribution for the User and Number of Hashtags with the Highest Proportion of Negative Comments
user_hashtag_sentiment_distribution.loc[[user_hashtag_with_most_negative_comments]].plot(kind='bar', stacked=True, ax=axs[0, 1])
axs[0, 1].set_title(f'Sentiment Distribution for User {user_hashtag_with_most_negative_comments[0]} Using {user_hashtag_with_most_negative_comments[1]} Hashtags')
axs[0, 1].set_xlabel('Sentiment')
axs[0, 1].set_ylabel('Percentage')
axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=0)
​
# Sentiment Distribution for the User and Number of Hashtags with the Highest Proportion of Neutral Comments
user_hashtag_sentiment_distribution.loc[[user_hashtag_with_most_neutral_comments]].plot(kind='bar', stacked=True, ax=axs[1, 0])
axs[1, 0].set_title(f'Sentiment Distribution for User {user_hashtag_with_most_neutral_comments[0]} Using {user_hashtag_with_most_neutral_comments[1]} Hashtags')
axs[1, 0].set_xlabel('Sentiment')
axs[1, 0].set_ylabel('Percentage')
axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(), rotation=0)
​
# Sentiment Distribution for the Most Frequent Commenters
top_commenting_users[['negative', 'neutral']].mean().plot(kind='pie', autopct='%1.1f%%', ax=axs[1, 1])
axs[1, 1].set_title('Sentiment Distribution for the Most Frequent Commenters')
axs[1, 1].set_ylabel('')
​
plt.tight_layout()
plt.show()
​

