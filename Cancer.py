
# coding: utf-8

# In[51]:


import math
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
print(X_train.shape,y_train.shape, X_test.shape, y_test.shape)


# In[53]:


def vis_data(X_train,y_train):
    plt.grid()
    
    for i in range(len(y_train)):
        if y_train[i]==1:
            color='red'
        elif y_train[i]==0:
            color='blue'
        plt.scatter(X_train[i][0],X_train[i][1],c=color)
    
vis_data(X_train,y_train)


# In[54]:


vis_data(X_test,y_test)


# In[55]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(metrics.accuracy_score(y_test, predictions))


# In[56]:


print(predictions)


# In[57]:


print(y_test)


# In[58]:


print (metrics.classification_report(y_test, predictions))
print (metrics.confusion_matrix(y_test, predictions))

