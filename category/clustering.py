from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import string
import pandas as pd

# nltk.download('punkt')
# nltk.download('wordnet')

document_df = pd.DataFrame({
        'opinion_text': [
            'localhost sshd Invalid user from port',
            'localhost sshd Received disconnect from port disconnected by user',
            'localhost sshd input_userauth_request:invalid user preauth',
            'localhost sshd Disconnected from port',
            'localhost sshd Connection closed by port preauth',
            'localhost sshd pam_unix session session closed for user ec2 user',
            'localhost sshd pam_unix session session opened for user ec2 user by uid',
            'localhost runuser pam_unix session session closed for user ec2 user',
            'localhost runuser pam_unix session session opened for user ec2 user by ec2 user uid'
        ]
    })



remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

tokenized_corpus = []
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

for text in document_df['opinion_text']:
    # print(text)
    tokenized_corpus.append(LemNormalize(text))

tfidf_vect = TfidfVectorizer(tokenized_corpus,
                            #stop_words='english',
                            ngram_range=(1,2))
                            # min_df=0.05, max_df=0.85)

# tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize,
#                             #stop_words='english',
#                             ngram_range=(1,2))
#                             # min_df=0.05, max_df=0.85)

ftr_vect = tfidf_vect.fit_transform(document_df['opinion_text'])

# print(tfidf_vect.idf_)

print(f"feature names:\n{tfidf_vect.get_feature_names()}")
# print(f"shape: {ftr_vect.shape}")
# print(f"feature names:\n{ftr_vect}")
# print(f"1-doc:\n{ftr_vect[0]}")
# print(f"1-doc values:\n{ftr_vect[0].toarray()}")

# K-means로 3개 군집으로 문서 군집화시키기
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6, max_iter=10000, random_state=42)
# 비지도 학습이니 feature로만 학습시키고 예측
cluster_label = kmeans.fit_predict(ftr_vect)

# 군집화한 레이블값들을 document_df 에 추가하기
document_df['cluster_label'] = cluster_label
print(document_df.sort_values(by=['cluster_label']))