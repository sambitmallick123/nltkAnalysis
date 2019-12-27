import nltk

from nltk.tokenize import sent_tokenize


text="""Hello everyone, how are you doing today? The weather is great, and let's start with today's seminar. Hello ! You shouldn't sleep during The presentation"""

tokenized_text=sent_tokenize(text)
print(tokenized_text)

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)


print("Frequency Distribution of tokenized words :")
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist.most_common(10))


#import matplotlib.pyplot as plt
#fdist.plot(30,cumulative=False)
#plt.show()

from nltk.corpus import stopwords
stop_words={'their', 'then', 'not', 'ma', 'here', 'other', 'won', 'up', 'weren', 'being', 'we', 'those', 'an', 'them', 'which', 'him', 'so', 'yourselves', 'what', 'own', 'has', 'should', 'above', 'in', 'myself', 'against', 'that', 'before', 't', 'just', 'into', 'about', 'most', 'd', 'where', 'our', 'or', 'such', 'ours', 'of', 'doesn', 'further', 'needn', 'now', 'some', 'too', 'hasn', 'more', 'the', 'yours', 'her', 'below', 'same', 'how', 'very', 'is', 'did', 'you', 'his', 'when', 'few', 'does', 'down', 'yourself', 'i', 'do', 'both', 'shan', 'have', 'itself', 'shouldn', 'through', 'themselves', 'o', 'didn', 've', 'm', 'off', 'out', 'but', 'and', 'doing', 'any', 'nor', 'over', 'had', 'because', 'himself', 'theirs', 'me', 'by', 'she', 'whom', 'hers', 're', 'hadn', 'who', 'he', 'my', 'if', 'will', 'are', 'why', 'from', 'am', 'with', 'been', 'its', 'ourselves', 'ain', 'couldn', 'a', 'aren', 'under', 'll', 'on', 'y', 'can', 'they', 'than', 'after', 'wouldn', 'each', 'once', 'mightn', 'for', 'this', 'these', 's', 'only', 'haven', 'having', 'all', 'don', 'it', 'there', 'until', 'again', 'to', 'while', 'be', 'no', 'during', 'herself', 'as', 'mustn', 'between', 'was', 'at', 'your', 'were', 'isn', 'wasn'}

#print(stop_words)


filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
#print("Tokenized Sentence:",tokenized_word)
print("Filterd Sentence:",filtered_sent)


# Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))

#print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

#Lexicon Normalization
#performing stemming and Lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

#word = "SSES"
#print("Lemmatized Word:",lem.lemmatize(word,"v"))
#print("Stemmed Word:",stem.stem(word))





########################## Part 2

print("")
print("Part 2")
#sent = "Albert Einstein was born in Ulm, Germany in 1879."

sent = "I am really enjoying Paderborn."
tokens=nltk.word_tokenize(sent)
print(tokens)
nltk.pos_tag(tokens)


import pandas as pd

data=pd.read_csv('train.tsv', sep='\t')

#print(data.head())
#print(data.info())

data.Sentiment.value_counts()

#Sentiment_count=data.groupby('Sentiment').count()
#plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
#plt.xlabel('Review Sentiments')
#plt.ylabel('Number of Review')
#plt.show()


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['Sentiment'], test_size=0.3, random_state=1)


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes

clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("Model Accuracy:",metrics.accuracy_score(y_test, predicted))


######### END

#from sklearn.feature_extraction.text import TfidfVectorizer
#tf=TfidfVectorizer()
#text_tf= tf.fit_transform(data['Phrase'])

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(
#    text_tf, data['Sentiment'], test_size=0.3, random_state=123)

#from sklearn.naive_bayes import MultinomialNB
#from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
#clf = MultinomialNB().fit(X_train, y_train)
#predicted= clf.predict(X_test)
#print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


