import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings('ignore')
from numpy.linalg import norm

     

S1 = "sunshine state enjoy sunshine"
S2 = "brown fox jump high, brown fox run"
S3 = "sunshine state fox run fast"
     

CountVec = CountVectorizer(ngram_range=(1,1))

Count_data = CountVec.fit_transform([S1,S2,S3])
cv_data = Count_data.toarray()

cv_dataframe=pd.DataFrame(cv_data,columns=CountVec.get_feature_names())
print(cv_dataframe)

     
#TF Model
tf_vec = TfidfVectorizer(use_idf=False)
tf_data = tf_vec.fit_transform([S1, S2, S3])
tf= np.round_(tf_data.toarray(),decimals = 3)

tf_dataframe=pd.DataFrame(tf,columns=tf_vec.get_feature_names())
tf_dataframe

     

#TF-IDF Model
tf_idf_vec = TfidfVectorizer(use_idf=True) 
tf_idf_data = tf_idf_vec.fit_transform([S1, S2, S3])
tfidf_data=tf_idf_data.toarray()
tf_idf_dataframe=pd.DataFrame(tfidf_data,columns=tf_idf_vec.get_feature_names())
tf_idf_dataframe
     
t1 = np.array(tfidf_data[0])
t3 = np.array(tfidf_data[2])
     

# compute cosine similarity
cosine = np.dot(t1,t3)/(norm(t1)*norm(t3))
print("Cosine Similarity between S1 & S3:", round(cosine,2))

     
