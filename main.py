from operator import ne
import numpy as np  # creating array
import pandas as pd # creating dataframes and storing data in that dataframes
import re  # regular expression (very useful for searching the text in the document)
from nltk.corpus import stopwords  # nltk = natural language tool kit, corpus = body or content of particular text
from nltk.stem.porter import PorterStemmer # PorterStemmer removes infix and suffix of particular word and give root word
from sklearn.feature_extraction.text import TfidfVectorizer # to convert text into feature vector (number)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#print(stopwords.words('english')) #printing stopwords, stopwords are the words which doesn't add any value to context

news_dataset = pd.read_csv('train.csv')

#print(news_dataset.shape) # (rows, column) (20800, 5)

#print(news_dataset.head()) # print first 5 rows of dataframe

#print(news_dataset.isnull().sum()) # counting the number of missing values in dataset

news_dataset = news_dataset.fillna('') # replacing the null values with empty string

news_dataset['content'] = news_dataset['author']+ ' ' + news_dataset['title'] # merging the author name and news title in new column 'content'

#print(news_dataset['content'])


# SEPERATING THE DATA AND LABEL

x = news_dataset.drop(columns='label', axis=1) # removing label column and put the remaining features in 'x' variable
y = news_dataset['label'] # variable 'y' points to label column


# STEMMING : THE PROCESS OF REDUCING A WORD TO ITS ROOT WORD
# eg. actor, actress, acting --> act

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content) # here sub=substitute, exclude everything in content which is other than alphabet and things apart from alphabets are replaced by space character
    stemmed_content = stemmed_content.lower() # convert content into lower case
    stemmed_content = stemmed_content.split() # all words are splited and converted into list
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # reduce word into its root word and remove stopwords
    stemmed_content = ' '.join(stemmed_content) # now join all the words
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming) # calling stemming function by passing content column

#print(news_dataset['content'])

x = news_dataset['content'].values
y= news_dataset['label'].values





# CONVERTING THE TEXTUAL DATA INTO NUMERICAL DATA

# Tf = term frequency  and  idf = inverse document frequency
# Tf = count the no. of times the particular word is repeated in a document, so it tells the model that it is a very important word and assign numerical value to that word.
# idf = count that repeated word in a content that doesn't add any value to content and reduce its importance

vectorizer = TfidfVectorizer() 
vectorizer.fit(x) # x is fitted in vectorizer
x = vectorizer.transform(x) # now all the words in x are converted into their respected numerical values

#print(x)






# SPLITTING THE DATASET TO TRAINING AND TEST DATA

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# stratify = y  means real news and fake news will be segregated into equal proportion as 'y' contains the label
# random_state = 3  means data will be splitted in same way in future as this is splitting for me now (you can use any number in random_state, it is basically to reproduce particular code)









# TRAINING THE MODEL : LOGISTIC REGRESSION

model = LogisticRegression()
model.fit(x_train, y_train)



# ACCURACY SCORE ON TRAINING DATA

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

#print('Accuracy score of training data : ', training_data_accuracy)






# ACCURACY SCORE ON TEST DATA

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

#print('Accuracy score of test data : ', test_data_accuracy)







# MAKING A PREDICTIVE SYSTEM

x_new = x_test[3] # taking third row
prediction = model.predict(x_new)
print(prediction)

if(prediction[0]==0):
    print('The news is real')
else:
    print('The news is fake')


print(y_test[3])

