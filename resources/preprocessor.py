from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

'''
Turns text labels into numbers
'''
def encode_labels(news):
    encoder=LabelEncoder()
    news['label']=encoder.fit_transform(news['label'])

# Vectorisers
'''
Tokenise and aplies Bag of Words vectorization
'''
def bag_of_words_vectorize(news):
    cv = CountVectorizer(max_df = 0.7, stop_words = 'english')
    X = cv.fit_transform(news['tweet'])
    y = news.label.values
    return X,y

'''
Tokenise and aplies IFIDF vectorization
'''
def tfidf_vectorize(news):
    tfidf = TfidfVectorizer(max_df = 0.7, stop_words = 'english')
    X = tfidf.fit_transform(news['tweet'])
    y = news.label.values
    return X,y