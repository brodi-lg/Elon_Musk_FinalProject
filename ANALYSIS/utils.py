# utils.py

##has all functions except for the BERT 

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from detoxify import Detoxify


# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

def analyze_comment_length(df, text_column='Comment'):
    
    """Function to analyze comments in terms of characters and word count.
    Adds comment length and word count."""
    
    # calculate length of comment in characters
    df['Comment_Length'] = df[text_column].apply(len)
    
    # calculate word count of comment
    df['Word_Count'] = df[text_column].apply(lambda x: len(str(x).split()))
    
    #results 
    print("Basic Statistics for Comment Length and Word Count:")
    print(df[['Comment_Length', 'Word_Count']].describe())
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Comment_Length'], bins=30, kde=True, color='blue')
    plt.title('Distribution of Comment Length (Characters)')
    plt.xlabel('Length (Characters)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df['Word_Count'], bins=30, kde=True, color='green')
    plt.title('Distribution of Word Count')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return df


# VADER Sentiment Analysis Function
def analyze_vader_sentiment(text):

    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return compound, sentiment

# TextBlob Sentiment Analysis Function
def analyze_textblob_sentiment(text):
   
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return polarity, sentiment

# Process Comments and Apply Sentiment Analysis
def process_comments(df):
    
    df[['VADER_Score', 'VADER_Sentiment']] = df['Comment'].apply(
        lambda x: pd.Series(analyze_vader_sentiment(x)))
    df[['TextBlob_Score', 'TextBlob_Sentiment']] = df['Comment'].apply(
        lambda x: pd.Series(analyze_textblob_sentiment(x)))
    
    return df


### Visualization Functions after processing with vader and textblob
def plot_sentiment_distribution(df, title):
   
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df['VADER_Sentiment'], label='VADER', fill=True, color='blue')
    sns.kdeplot(df['TextBlob_Sentiment'], label='TextBlob', fill=True, color='red')
    plt.title(title)
    plt.legend()
    plt.show()

def scatter_vader_vs_textblob(df, title):
   
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['VADER_Sentiment'], y=df['TextBlob_Sentiment'])
    plt.xlabel("VADER Sentiment Score")
    plt.ylabel("TextBlob Sentiment Score")
    plt.title(title)
    plt.show()


def plot_sentiment_comparison_boxplot(df, title):
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[['VADER_Sentiment', 'TextBlob_Sentiment']], palette='Set2')
    plt.title(title)
    plt.ylabel("Sentiment Score")
    plt.xticks([0, 1], ['VADER', 'TextBlob'], rotation=0)
    plt.show()
    

#wordcloud analysis
def generate_filtered_wordcloud(df, column, title, extra_stopwords=None):
   
    text = " ".join(df[column].dropna().astype(str))
    
    # Define default stopwords and add custom ones
    stopwords = set(STOPWORDS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)
    
    # Generate word cloud with filtering
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap ="plasma",max_words =30, contour_color = "red",
                        
                          stopwords=stopwords, collocations=False).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

###visualisation functions for BERT Analysis


def plot_bert_pie_chart(df, title="BERT Sentiment Distribution"):
    
    sentiment_counts = df['BERT_Sentiment'].value_counts()

    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    plt.title(title)
    plt.ylabel("")  # Hide y-axis label
    plt.show()

def plot_bert_bar_chart(df, title="BERT Sentiment Score Distribution"):
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df['BERT_Score'], bins=10, kde=True, color='purple')
    plt.xlabel("Sentiment Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()

def plot_bert_violin_plot(df, title="BERT Sentiment Score Distribution"):
    
    plt.figure(figsize=(8, 5))
    sns.violinplot(y=df['BERT_Score'], color='green')
    plt.ylabel("Sentiment Score")
    plt.title(title)
    plt.show()


### comparing the SAME set of comments with vader, textblob and BERT

# Load BERT Sentiment Analysis Pipeline 
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Apply BERT Sentiment Analysis
def analyze_bert_sentiment(text):
    result = sentiment_pipeline(text[:512])[0]  # Truncate to max 512 tokens
    return result['label'], result['score']

# Apply VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()
def analyze_vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    sentiment = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")
    return sentiment, compound

# Apply TextBlob Sentiment Analysis
def analyze_textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    sentiment = "Positive" if polarity > 0 else ("Negative" if polarity < 0 else "Neutral")
    return sentiment, polarity

# Function to compare sentiment analysis using BERT, VADER, and TextBlob
def compare_sentiments(df, comment_column='Comment', n=10):
   
    small_batch = df.head(n)  # Adjust number if needed

    # Ensure Comment column is present
    if comment_column not in small_batch.columns:
        raise ValueError(f"The dataset must have a column named '{comment_column}'.")

    # Apply sentiment analysis for each model
    small_batch[['BERT_Sentiment', 'BERT_Score']] = small_batch[comment_column].apply(lambda x: pd.Series(analyze_bert_sentiment(str(x))))
    small_batch[['VADER_Sentiment', 'VADER_Score']] = small_batch[comment_column].apply(lambda x: pd.Series(analyze_vader_sentiment(str(x))))
    small_batch[['TextBlob_Sentiment', 'TextBlob_Score']] = small_batch[comment_column].apply(lambda x: pd.Series(analyze_textblob_sentiment(str(x))))

    # Visualization: Comparison of Sentiments for BERT, VADER, and TextBlob
   
    sentiment_comparison = pd.melt(small_batch[[comment_column, 'BERT_Sentiment', 'VADER_Sentiment', 'TextBlob_Sentiment']],
                                   id_vars=[comment_column], value_vars=['BERT_Sentiment', 'VADER_Sentiment', 'TextBlob_Sentiment'],
                                   var_name="Model", value_name="Sentiment")

    # Visualization: Bar plot of Sentiment Comparison
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Sentiment', hue='Model', data=sentiment_comparison, palette='Set2')

    # Customize title and labels
    plt.title('Sentiment Comparison: BERT vs VADER vs TextBlob', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=14)
    plt.ylabel('Number of Comments', fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)

    # Show plot
    plt.tight_layout()
    plt.show()

    # Return the results DataFrame
    return small_batch

