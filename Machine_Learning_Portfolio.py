#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd



# In[3]:


df=pd.read_csv("C:\\Users\\HP\\Downloads\\Indian Liver Patient Dataset (ILPD).csv")
print(df.head())
print(df.shape)


# In[4]:


#Checking for not nulls
df.isnull().sum()


# In[5]:


X = df.dropna(subset=['alkphos'])
y = X['is_patient']
print(y.shape)
print(X.shape)


# In[6]:


display(df.describe())


# In[7]:


X['is_patient'].value_counts()


# In[8]:


X.loc[:,'gender'] = X['gender'].apply(lambda x:1 if x == 'Male' else 0)


# In[9]:


print(X['gender'])


# In[10]:


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sklearn.datasets as dt
import sklearn.model_selection as ms
import sklearn.neighbors as ne 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[11]:


X.dropna()
X = X.iloc[:, :10]


# In[12]:


scaler=MinMaxScaler()
scaled_values=scaler.fit_transform(X)
X.loc[:,:]=scaled_values
print(X)


# In[13]:


X.isnull().sum()


# In[14]:


nan_values = X.isna().sum()
print(nan_values)


# In[15]:


X_new = X.dropna(subset=['alkphos'])
print(y.shape)
print(X_new.shape)


# In[16]:


print(X_new)


# In[17]:


#set seed for reproducibility
SEED=42

# Instantiate lr
lr = LogisticRegression(random_state=SEED)

# Instantiate knn
knn = KNN(n_neighbors=27)
print(knn)
# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Decision Tree', dt)]


# In[ ]:





# In[18]:


nan_values = X_new.isnull().sum()
print(nan_values)
#column_name = 'alkphos'
#X_new = X.dropna(subset=['alkphos'],inplace = True)


# In[19]:


print(X_new)
#nan_values = X_new.isna().sum()
#print(nan_values)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=SEED)
print(y_train)
print(y_test)


# In[21]:


# Decision tree

#Create a Decision Tree classifier with the current max_depth

from sklearn import tree

DT=tr.DecisionTreeClassifier(max_depth=3)

DT.fit(X_train,y_train)

trACC=DT.score(X_train,y_train)

tesACC=DT.score(X_test,y_test)
 
#print(trACC)

#print(tesACC)

model = DT.fit(X_train, y_train)

text_representation = tr.export_text(DT)

print(text_representation)
 
# To Visualize Decision Tree

tree.plot_tree(model)


# In[22]:


import matplotlib.pyplot as plt
# Decision tree
trACC=[]
tesACC=[]
MD=[]

for i in range(1,8):
    #Create a Decision Tree classifier with the current max_depth
    DT=tr.DecisionTreeClassifier(max_depth=i)
    DT.fit(X_train,y_train)
    trACC.append(DT.score(X_train,y_train))
    tesACC.append(DT.score(X_test,y_test))
    MD.append(i)
#print(trACC)
#print(tesACC)
#print(MD)
plt.figure()
plt.plot(MD, trACC, label='Train',marker='o')
plt.plot(MD, tesACC, label='Test', marker='o')
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#print(trACC)
#print(tesACC)


# In[23]:


best_score = 0.0
best_k = 0
for k in range(1, 25):
    knn_clf_sk = KNN(n_neighbors=k)
    knn_clf_sk.fit(X_train, y_train)
    score = knn_clf_sk.score(X_test, y_test)
    if score > best_score:
        best_k = k
        best_score = score

print("best_k = " + str(best_k))
print("best_score = " + str(best_score))
    


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
 
# Assuming you have your data in X_train, X_test, y_train, y_test
test_score=[]
train_score=[]
MD=[]
for i in range(1,25):
    knn=KNN(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_score.append(accuracy_score(y_train, y_pred_train))
    test_score.append(accuracy_score(y_test, y_pred_test))
    MD.append(i)
print(test_score)
print(test_score)
print(MD)  
# Visulaize ACC
plt.figure()
plt.plot(MD, train_score, label='Train',marker='o')
plt.plot(MD, test_score, label='Test',marker='o')
plt.xlabel('K_neighbors')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()


# In[35]:


# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

LR=LogisticRegression(max_iter=200)
LR.fit(X_train,y_train)

ACC_tr=LR.score(X_train,y_train)
ACC_tes=LR.score(X_test,y_test)


print('Train Accuracy',ACC_tr)
print('Test Accuracy', ACC_tes)


# In[39]:


# Confusion matrix
y_pred = LR.predict(X)
print(y_pred)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))


# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt


cm = confusion_matrix(y, y_pred)


# Define the labels for the confusion matrix
labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
 
# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[61]:


# Iterate over the pre-defined list of classifiers
for clf_name, clf in classifiers:    
 
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
   
    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) 
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))


# In[ ]:




