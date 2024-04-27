import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

df = pd.read_csv('/kaggle/input/amazon-reviews/trimmed_reviews.csv')

ax = df['overall'].value_counts().sort_index().plot(kind = 'bar', title = 'Count of reviews by stars', figsize = (10, 5))
ax.set_xlabel('Review Stars')
plt.show()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row['reviewText'])
    myid = i
    res[myid] = sia.polarity_scores(text)
    

ax = sns.barplot(data = vaders, x = 'overall', y = 'compound')
ax.set_title("VADER score vs amazon reviews")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(vaders['overall'], vaders['compound'], color = "blue", alpha = 0.05)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(vaders['overall'], vaders['pos'], color = "blue", alpha = 0.05)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='overall', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='overall', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='overall', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax