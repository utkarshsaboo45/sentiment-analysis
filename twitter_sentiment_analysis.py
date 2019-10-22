import pandas as pd 
import re
from wordcloud import WordCloud
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers

TRAIN_SOURCE = 'tweets_train.csv'
PREPROCESSED = 'train_test_set_preprocessed.csv'
CLEAN = 'cleaned_tweet'
LABEL = 'label'

train_set = pd.read_csv(TRAIN_SOURCE)
train_test_set = pd.read_csv(PREPROCESSED)

#Wordcloud
words = ' '.join([text for text in train_test_set[CLEAN]])
wc = WordCloud(width = 800, height = 500, 
                      random_state = 21, max_font_size = 110).generate(words)
plt.figure(figsize = (10, 7))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

negative_words = ' '.join([text for text in train_test_set[CLEAN][train_test_set[LABEL] == 1]])
wc = WordCloud(width = 800, height = 500, 
                      random_state = 21, max_font_size = 110).generate(negative_words)
plt.figure(figsize = (10, 7))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

non_negative_words =' '.join([text for text in train_test_set[CLEAN][train_test_set[LABEL] == 0]])
wc = WordCloud(width = 800, height = 500, 
                      random_state = 21, max_font_size = 110).generate(non_negative_words)
plt.figure(figsize = (10, 7))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#Extracting Hashtags
def extract_hashtag(tweets):
    hashtags = []
    for tweet in tweets:
        hashtag = re.findall(r'#(\w+)', tweet)
        hashtags.append(hashtag)
    return hashtags

regular_hashtags = sum(extract_hashtag(train_test_set[CLEAN][train_test_set[LABEL] == 0]), [])
negative_hashtags = sum(extract_hashtag(train_test_set[CLEAN][train_test_set[LABEL] == 1]), [])

#Plotting top Positive Hashtags based on frequency
temp = FreqDist(regular_hashtags)
positive_hashtag_distribution = pd.DataFrame({'Hashtag': list(temp.keys()),
                  'Hashtag_count': list(temp.values())})
positive_hashtag_distribution = positive_hashtag_distribution.nlargest(columns = "Hashtag_count", n = 10)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = positive_hashtag_distribution, x = "Hashtag", y = "Hashtag_count")
ax.set(ylabel = 'Hashtag Count')
plt.show()

#Plotting Negative Hashtags based on frequency
temp = FreqDist(negative_hashtags)
negative_hashtag_distribution = pd.DataFrame({'Hashtag': list(temp.keys()), 
                  'Hashtag_count': list(temp.values())})
negative_hashtag_distribution = negative_hashtag_distribution.nlargest(columns="Hashtag_count", n = 10)   
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = negative_hashtag_distribution, x = "Hashtag", y = "Hashtag_count")
ax.set(ylabel = 'Hashtag Count')
plt.show()

#TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, min_df = 2, max_features = 1000, stop_words = 'english')
tfidf = tfidf_vectorizer.fit_transform(train_test_set[CLEAN])

train = tfidf[:len(train_set),:]
test = tfidf[len(train_set):,:]

xtrain, xval, ytrain, yval = train_test_split(train, train_set[LABEL], random_state = 1000, test_size = 0.25)

xtrain_tfidf = train[ytrain.index]
xtest_tfidf = train[yval.index]

#Artificial Neural Netwoek
input_dim = xtrain.shape[1]

analyzer = Sequential()
analyzer.add(layers.Dense(10, input_dim = input_dim, activation = 'relu'))
analyzer.add(layers.Dense(1, activation = 'sigmoid'))

analyzer.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
analyzer.summary()

history = analyzer.fit(xtrain, ytrain, 
                    epochs = 100, 
                    validation_data = (xval, yval), 
                    batch_size = 10)

analyzer.save('sentiment_analysis_model.h5')

loss_train, accuracy_train = analyzer.evaluate(xtrain, ytrain)
loss_test, accuracy_test = analyzer.evaluate(xval, yval)
loss, accuracy = analyzer.evaluate(test)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)