#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import json
from sklearn import model_selection
import string
from math import log
from copy import deepcopy
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
#Initializing Stop words
stop_words = set(stopwords.words("english"))


# ### Initializing valid set of Categories

# In[25]:


#Set of valid categories : 
#Business, Comedy, Sports, Crime, Religion, Healthy Living, Politics.
category_dict={'business':0,'comedy':0,'sports':0,'crime':0,'religion':0,'healthy living':0,'politics':0}


# In[26]:


#news_category_dataset.json
df=pd.read_json('news_category_dataset.json',lines=True)


# In[27]:


df.head()


# In[28]:


df=df.loc[:, ['headline', 'category']]
df.head()


# In[29]:


#Extracting data in the valid set of categories
df=df[df['category'].str.lower().isin(category_dict.keys())]
df.head()


# In[30]:


df.shape


# ### Test - Train Split

# In[31]:


#Independent Variable
X=df['headline'].to_numpy()
#Dependent Variable
Y=df['category'].to_numpy()
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.15, random_state=0)
print(X_train.shape[0])


# In[32]:


print(Y_test.shape[0])


# ### Multinomial Naive Bayes Classifier

# In[33]:


class NB_Multinomial:
    
    def __init__(self):
        #Stores the Vocabulary and class wise doc count for each word
        self.word_class_dict={}
        self.prior=deepcopy(category_dict)
        self.cond_probablity={}
        #Calculates number of words per class
        self.class_words=deepcopy(category_dict)
        
    def preprocess(self,X,Y):
        global category_dict
        word_count={}
        #Creating the Word Class Dictionary
        for i in range(len(X)):
            _category=Y[i].lower()
            #Removing Punctuation and Converting to lower case
            temp_str=X[i].translate(str.maketrans('', '', string.punctuation)).lower()
            word_tokens = temp_str.split(' ')
            #Removing stopwords and adding to vocabulary
            for w in word_tokens:
                if w not in stop_words and len(w)>1:
                    #Counting Word Frequency
                    if w not in word_count.keys():
                        word_count[w]=0
                    else:
                        word_count[w]+=1
                        
                    word_in_dict=w in self.word_class_dict.keys()
                    if not word_in_dict:
                        self.word_class_dict[w]=deepcopy(category_dict)
                        self.word_class_dict[w][_category]+=1
                        self.class_words[_category]+=1
                    elif word_in_dict:
                        self.word_class_dict[w][_category]+=1
                        self.class_words[_category]+=1
        
        #Building Vocabulary
        for w in word_count.keys():
            if word_count[w]<2:
                #Deleting word not elligible to be present in dictionary
                del self.word_class_dict[w]
        
    def convert_to_lower(self,Y):
        total_len=len(Y)
        target=[]
        for i in range(total_len):
            target.append(Y[i].lower())
        return target
    
    def multinomial_probability(self,X,Y):
        global category_dict
        #Calculating Prior Probablity Values
        total_docs=len(Y)
        #Converting Y to lower case
        target=self.convert_to_lower(Y)
        docs_in_class=deepcopy(category_dict)
        
        #No. of headlines(documents) in each category
        for cat in target:
            docs_in_class[cat]+=1
        
        total_vocab_size=len(self.word_class_dict.keys())
        #Calculating Prior probability for each category
        for category in self.prior.keys():
            self.prior[category]=docs_in_class[category]/total_docs
        #Calculating Conditional Probablity for each category
        for w in self.word_class_dict.keys():
            self.cond_probablity[w]=deepcopy(category_dict)
            for category in self.cond_probablity[w].keys():
                self.cond_probablity[w][category]=(self.word_class_dict[w][category]+1)/(self.class_words[category]+total_vocab_size)
        
    
    def fit_transform(self,X,Y):
        self.preprocess(X,Y)
        self.multinomial_probability(X,Y)
    
    def extract_terms(self,D):
        vocab=[]
        #Removing punctuation and converting to lower case
        temp_str=D.translate(str.maketrans('', '', string.punctuation)).lower()
        word_tokens = temp_str.split(' ')
        #Removing Stop Words
        for w in word_tokens:
            if w not in stop_words and len(w)>1:
                vocab.append(w)
        
        return vocab
    
    def predict(self,X):
        global category_dict
        Y_pred=[]
        total_vocab_size=len(self.word_class_dict.keys())
        for i in range(len(X)):
            class_score=deepcopy(category_dict)
            #Extracting vocabulary for every (headline)document
            doc_vocab=self.extract_terms(X[i])
            for category in class_score.keys():
                class_score[category]=log(self.prior[category])
                for w in doc_vocab:
                    if w in self.word_class_dict.keys():
                        class_score[category]+=log(self.cond_probablity[w][category])
                    else:
                        #Conditional Probablity for a word not present in trained vocabulary
                        class_score[category]+=log(1/(self.class_words[category]+total_vocab_size))
            #Finding the category having maximum score
            predicted_category=max(class_score, key=class_score.get)
            Y_pred.append(predicted_category.upper())
        
        return Y_pred
    
    def accuracy_score(self,Y_pred,Y_test):
        target_pred=Y_pred
        target_test=Y_test
        count=0
        for i in range(len(target_pred)):
            if target_pred[i]==target_test[i]:
                count+=1
        acc_score=count/len(target_pred)
        print("Accuracy of the model is %s%%"%(round(acc_score*100,2)))
        


# ### Creating Classifier Model and Training 

# In[34]:


model=NB_Multinomial()
start_time=time.time()
model.fit_transform(X_train,Y_train)
time_to_train=time.time()-start_time


# ### Model Training and Preprocessing Time

# In[35]:


print("Time taken to train and preprocess model is : %s seconds "%(round(time_to_train,2)))


# In[36]:


Y_pred=model.predict(X_test)


# ### Overall Test Accuracy of Model 

# In[37]:


model.accuracy_score(Y_pred,Y_test)


# In[38]:


print(classification_report(Y_pred, Y_test))


# ### Confusion Matrix

# In[39]:


category_list=list(category_dict.keys())
category_list=[x.upper() for x in category_list]
cm = confusion_matrix(Y_test, Y_pred,labels=category_list)
print(cm)


# ### Class Wise Accuracies

# In[40]:


#Normalizing the diagonal entries
cm_norm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
class_accuracy=cm_norm.diagonal()*100
print("------------Class Wise Accuracies------------\n")
for i in range(len(category_list)):
    print(category_list[i]+" Accuracy = %s%% "%(round(class_accuracy[i],2)))


# In[41]:


fig = plt.figure(figsize = (14, 5)) 
# creating the bar plot 
class_accuracy=[round(x,2) for x in class_accuracy]
plt.bar(category_list, class_accuracy, color ='navy',  
        width = 0.4) 
  
plt.xlabel("Categories") 
plt.ylabel("Accuracy Percentage") 
plt.title("Accuracy of different Categories") 
plt.show() 


# ### Confusion Matrix with Category labels

# In[42]:


df_cm= pd.DataFrame(
    cm, 
    index=category_list, 
    columns=category_list
)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,fmt='d')

