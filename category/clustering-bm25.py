import pandas as pd

document_df = pd.DataFrame({
        'opinion_text': [
            'localhost sshd Invalid user from port',
            'localhost sshd Received disconnect from port disconnected by user',
            'localhost sshd input_userauth_request invalid user preauth',
            'localhost sshd Disconnected from port',
            'localhost sshd Connection closed by port preauth',
            'localhost sshd pam_unix session session closed for user ec2 user',
            'localhost sshd pam_unix session session opened for user ec2 user by uid',
            'localhost runuser pam_unix session session closed for user ec2 user',
            'localhost runuser pam_unix session session opened for user ec2 user by ec2 user uid'
        ]
    })


import string
import nltk
from nltk.stem import WordNetLemmatizer

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


tokenized_corpus = []

for text in document_df['opinion_text']:
    # print(text)
    tokenized_corpus.append(LemNormalize(text))

# print(tokenized_corpus)

from rank_bm25 import BM25Okapi
import numpy as np

def get_feature_names(bm25):
    return list(bm25.idf.keys())

bm25 = BM25Okapi(tokenized_corpus)
# print(bm25)
# print(bm25.corpus_size)
# print(bm25.idf)
# print(bm25.doc_len)
print(f"feature_names:\n{get_feature_names(bm25)}")

def match_term_value(idf, term, query):
    for q in query:
        if q == term:
            return idf.get(term)
    return 0

def get_bm25_weights(bm25, corpus):
    term_size = len(bm25.idf)
    print(f"term size: {term_size}")

    term_weight = np.empty((0, term_size), dtype=np.float64)    
    for query in corpus:
        weigth = np.array([(match_term_value(bm25.idf, term, query) or 0) for term in bm25.idf])
        weigth = weigth.reshape((1,-1))
        term_weight = np.append(term_weight, weigth, axis=0)

    
    return term_weight

ftr_vect = get_bm25_weights(bm25, tokenized_corpus)


# K-means로 3개 군집으로 문서 군집화시키기
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, max_iter=10000, random_state=42)
# 비지도 학습이니 feature로만 학습시키고 예측
cluster_label = kmeans.fit_predict(ftr_vect)

# 군집화한 레이블값들을 document_df 에 추가하기
document_df['cluster_label'] = cluster_label
print(document_df.sort_values(by=['cluster_label']))