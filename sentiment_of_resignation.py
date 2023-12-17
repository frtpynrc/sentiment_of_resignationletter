# Import necessary libraries
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Read the text file
text = open('resignation.txt', encoding='utf-8').read()

# Convert the text to lowercase
lower_case = text.lower()

# Remove punctuation from the text
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Tokenize the cleaned text
tokenized_words = word_tokenize(cleaned_text, "english")

# Remove stop words from the tokenized words
final_words = []
for word in tokenized_words:
    if word not in stopwords.words('english'):
        final_words.append(word)

# Lemmatize the final words to convert plurals to singles and get the base form of words
lemma_words = []
for word in final_words:
    word = WordNetLemmatizer().lemmatize(word)
    lemma_words.append(word)

# Read and extract emotions from the emotions.txt file
emotion_list = []
with open('status.txt', 'r') as file:
    for line in file:
        # Clean each line and split into word and emotion
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        # Check if the word is in lemmatized words
        if word in lemma_words:
            emotion_list.append(emotion)

# Print the list of emotions and their counts
print(emotion_list)
w = Counter(emotion_list)
print(w)

# Function to perform sentiment analysis using NLTK's SentimentIntensityAnalyzer
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")

# Analyze the sentiment of the cleaned text
sentiment_analyse(cleaned_text)

# Plot a bar chart for the emotions
fig, ax1 = plt.subplots()
ax1.pie(w.values(), labels=w.keys(), autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('pie_chart.png')
plt.show()
