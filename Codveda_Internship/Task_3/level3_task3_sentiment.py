import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv("Data/Sentiment dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

# Use the real text column
text_column = "Text"

# Function to classify sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))

    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Predicted_Sentiment"] = df[text_column].apply(get_sentiment)

# Show results
print("\nPredicted sentiment counts:")
print(df["Predicted_Sentiment"].value_counts())

# Plot sentiment distribution
df["Predicted_Sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Charts/sentiment_distribution.png")
plt.close()

# Create word cloud
text = " ".join(df[text_column].dropna().astype(str))

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("Charts/wordcloud.png")
plt.close()

print("\nSentiment analysis completed successfully!")