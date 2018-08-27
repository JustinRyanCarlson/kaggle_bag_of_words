import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 


train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.columns.values)
print()

example1 = review_to_words(train["review"][0])
print(example1)
print()

# pre-process all reviews
clean_train_reviews = []

print("Starting training reviews cleaning")
for raw_review in train['review'].tolist():
    clean_train_reviews.append(review_to_words(raw_review))
    if (len(clean_train_reviews))%1000 == 0:
        print(len(clean_train_reviews))

vectorizer = CountVectorizer(analyzer="word", max_features=20000)
fitted_vectors = vectorizer.fit_transform(clean_train_reviews)
fitted_vectors = fitted_vectors.toarray()
print(vectorizer.get_feature_names())

clf = RandomForestClassifier(n_estimators=1000)
clf_fitted = clf.fit(fitted_vectors, train["sentiment"])

test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)
clean_test_reviews = []
print("Starting test reviews cleaning")
for raw_review in test['review'].tolist():
    clean_test_reviews.append(review_to_words(raw_review))
    if (len(clean_test_reviews))%1000 == 0:
        print(len(clean_test_reviews))

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

predictions = clf_fitted.predict(test_data_features)
output = pd.DataFrame(data={"id":test["id"], "sentiment_pred":predictions})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)