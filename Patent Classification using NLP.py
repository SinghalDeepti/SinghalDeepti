#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[10]:


import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv('D:/Machine Learning/Feynn Labs ML Internship/Patent Data of 5 classes.csv', encoding='latin-1')
df['IPCfirst']=df['IPC'].str[:1]
counter = Counter(df['IPCfirst'].tolist())
print(counter)
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(5))}
df = df[df['IPCfirst'].map(lambda x: x in top_10_varieties)]
print(df) 
description_list = df['Patent Title'].tolist()
varietal_list = [top_10_varieties[i] for i in df['IPCfirst'].tolist()]
varietal_list = np.array(varietal_list)
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(description_list)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print(x_train_tfidf)


train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.3)
clf = MultinomialNB().fit(train_x, train_y)
y_score = clf.predict(test_x)
print(y_score)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Naive Bayes Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
NBA=(n_right/float(len(test_y))) * 100

clf = svm.SVC(kernel='linear').fit(train_x, train_y)
y_score = clf.predict(test_x)
print(y_score)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1
        
print("SVC Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))
SVA=(n_right/float(len(test_y))) *100

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(train_x, train_y) 
y_pred = classifier.predict(test_x)
RFA=accuracy_score(test_y, y_pred)*100
print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))
print(accuracy_score(test_y, y_pred))



# In[ ]:





# In[11]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Naive Bayes', 'SVC', 'Random Forest']
CLASSIFIERS = [NBA,SVA,RFA]
ax.bar(langs,students)
plt.show()


# In[ ]:




