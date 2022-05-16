#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes

# ### 1. Importing libraries

# In[1]:


# Importing the important libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
import warnings
warnings.filterwarnings('ignore')


# ### 2. Importing data

# In[4]:


df_test=pd.read_csv('SalaryData_Test.csv')
df_test


# In[5]:


df_train=pd.read_csv('SalaryData_Train.csv')
df_train


# In[ ]:





# ### 3. Data understanding

# In[6]:


df_train.info()


# In[10]:


df_test.info()


# In[7]:


# Converting the categorical columns into integer

from sklearn.preprocessing import LabelEncoder
df_train = df_train.apply(LabelEncoder().fit_transform)
df_train.head()


# In[9]:


#Converting the categorical columns into integer
from sklearn.preprocessing import LabelEncoder
df_test = df_test.apply(LabelEncoder().fit_transform)
df_test.head()


# ### 4. Model Building

# In[11]:


X_train = df_train.drop(['education','relationship','native','maritalstatus','sex','race'], axis=1).values
Y_train = df_train['Salary'].values
print(np.unique(Y_train))
X_train


# In[12]:


X_test = df_test.drop(['education','relationship','native','maritalstatus','sex','race'],axis=1).values
Y_test = df_test['Salary'].values
print(np.unique(Y_test))
X_test


# ### 5. Model Training | Model Testing

# In[13]:


#training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb =  GaussianNB()
gnb.fit(X_train, Y_train)

# Making predictions on the testing set
y_pred = gnb.predict(X_test)

# Comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, y_pred)*100)


# In[14]:


# GB
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.hist(y_pred, bins = 10)
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# In[15]:


# Preparing a naive bayes model on training data set

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# In[16]:


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(X_train, Y_train)
classifier_mb.score(X_train, Y_train)
classifier_mb.score(X_test, Y_test)
predicted_result = classifier_mb.predict(X_train)
accuracy_train = np.mean(predicted_result == Y_train)
accuracy_train


# In[17]:


test_predict = classifier_mb.predict(X_test)
accuracy_test = np.mean(test_predict == Y_test)
accuracy_test


# In[18]:


# MB
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.hist(predicted_result, bins = 10)
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# In[19]:


# Gaussian Naive Bayes
classifier_gb = GB()
classifier_gb.fit(X_train, Y_train)
classifier_gb.score(X_train, Y_train)
classifier_gb.score(X_test, Y_test)
train_pred = classifier_gb.predict(X_train)
accuracy_train = np.mean(train_pred == Y_train)
accuracy_train


# In[20]:


test_pred = classifier_gb.predict(X_test)
accuracy_test = np.mean(test_pred == Y_test)
accuracy_test


# In[21]:


# GB
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.hist(train_pred, bins = 10)
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')


# In[ ]:




