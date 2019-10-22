import pandas as pd 
import re
from nltk.stem import PorterStemmer

TRAIN_SOURCE = 'tweets_train.csv'
TEST_SOURCE = 'tweets_test.csv'
SLANGS_SOURCE = 'slang.txt'
PREPROCESSED = 'train_test_set_preprocessed.csv'
CLEAN = 'cleaned_tweet'

#Importing the datasets
train_set = pd.read_csv(TRAIN_SOURCE)
test_set = pd.read_csv(TEST_SOURCE)
slangs = pd.read_csv(SLANGS_SOURCE, sep="`", error_bad_lines = False)

#method to remove user handles
def removeHandles(pattern, input_txt):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

#Preparing the dataset
train_test_set = train_set.append(test_set, ignore_index = True)
#train_test_set = pd.read_csv('train_test_set_5000_15000.csv')

train_test_set[CLEAN] = ''
for i in range(len(train_test_set)):
    train_test_set[CLEAN][i] = removeHandles("@[\w]*", train_test_set['tweet'][i])

#Removing special characters apart from hashtags
train_test_set[CLEAN] = train_test_set[CLEAN].str.replace("[^a-zA-Z0-9#_']", " ")

#Preparing slangs dictionary
slangs = slangs.set_index('Slag').T.to_dict('list')
slangs =  {str(k).lower(): v for k, v in slangs.items()}
for i in slangs:
    slangs[i] = str(slangs[i][0]).lower()

#Replacing slangs with proper words/phrases
for i in range(5000):
    for j in slangs:
        train_test_set[CLEAN][i] = train_test_set[CLEAN][i].replace(' ' + j + ' ', ' ' + slangs[j] + ' ')
        train_test_set[CLEAN][i] = train_test_set[CLEAN][i].replace('#' + j + ' ', '#' + slangs[j] + ' ')
    print(i)

#Removing the remaining digits
train_test_set[CLEAN] = train_test_set[CLEAN].str.replace("[^a-zA-Z#]", " ")

#Shortword removal
train_test_set[CLEAN] = train_test_set[CLEAN].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

#Tokenization
tokenized_train_test_set = train_test_set[CLEAN].apply(lambda x: x.split())
tokenized_train_test_set.head()

#Stemming
stemmer = PorterStemmer()
tokenized_train_test_set = tokenized_train_test_set.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_train_test_set.head()

for i in range(len(tokenized_train_test_set)):
    tokenized_train_test_set[i] = ' '.join(tokenized_train_test_set[i])

train_test_set[CLEAN] = tokenized_train_test_set

#Saving the preprocessed file
train_test_set.to_csv(PREPROCESSED, index = False)

