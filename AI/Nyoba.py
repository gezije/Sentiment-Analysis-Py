import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Preprocessing Text
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Sentiment Analysis using NLTK
def analyze_sentiment_nltk(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    sentiment = sentiment_scores['compound']
    
    if sentiment >= 0.05:
        return 'Positive'
    elif sentiment <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Sentiment Analysis using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# User Input
num_sentences = int(input("Enter the number of sentences: "))
sentences = []

for i in range(num_sentences):
    sentence = input(f"Enter sentence {i+1}: ")
    sentences.append(sentence)

for i, sentence in enumerate(sentences):
    processed_text = preprocess_text(sentence)
    sentiment_nltk = analyze_sentiment_nltk(processed_text)
    sentiment_textblob = analyze_sentiment_textblob(processed_text)

    print(f"\nSentence {i+1}: {sentence}")
    print("Sentiment (NLTK):", sentiment_nltk)
    print("Sentiment (TextBlob):", sentiment_textblob)