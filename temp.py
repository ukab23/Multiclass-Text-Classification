# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

df = pd.read_csv('Consumer_Complaints.csv')
df = df.head(50000)
df = df.iloc[:,[1,5]]
df['Consumer_complaint_narrative'] = df['Consumer complaint narrative']
df = df[pd.notnull(df['Consumer_complaint_narrative'])]
df['category_id'] = df['Product'].factorize()[0]

category_id_df = df[['Product','category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','Product']].values)

fig = plot.figure(figsize=(8,6))
df.groupby('Product').Consumer_complaint_narrative.count().plot.bar(ylim = 0)
plot.show()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
                        encoding='latin-1', ngram_range=(1,2),
                        stop_words='english')
features = tfidf.fit_transform(df.Consumer_complaint_narrative).toarray()
labels = df.category_id
features.shape

from sklearn.feature_selection import chi2
N = 2
for Product, category_id in sorted(category_to_id.items()):
    features_chi = chi2(features, labels == category_id)
    indices = np.argsort(features_chi[0])
    feature_name = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_name if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_name if len(v.split(' ')) == 2]
    print("# '{}':".format(Product))
    print(" .Most related unigram:\n. {}".format('\n.'.join(unigrams[-N:])))
    print(" .Most related bigram:\n. {}".format('\n.'.join(bigrams[-N:])))      

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(df['Consumer_complaint_narrative'], df['Product'], random_state = 0) 
countvect = CountVectorizer()
X_train_counts = countvect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

print(clf.predict(countvect.transform(["I have gotten 7 calls from AR Resources looking for a person who is not authorized to use our phone. I have called them three times and they told me they would take our number off their list. I keep getting calls so started to document. I received a call on XX/XX/XXXX at XXXX XXXX. I called them to report it, spoke with XXXX, who told me she removed it from the list at XXXX XXXX on XX/XX/XXXX. At XXXX XXXX on XX/XX/XXXX I received another call. I called back again, speaking to a very rude XXXX, who said she removed my name from the list. I asked to be connected to a supervisor and she connected me to XXXX, who did not pick up. I left a message asking her to contact me. XXXX told me they were a call center. Please stop the calls - 7 times I've been told I was being taken off the list. After 35 years, I'm considering changing my phone number - that is not right. This is harassment. Thank you."])))




