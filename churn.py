#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction

# ### Importing the data set

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("Telco-Customer-Churn.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# # Exploratory Data Analysis

# In[6]:


df.isna().sum().sum() #missing values in the data set


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.Churn.value_counts()


# In[10]:


df.info()


# In[11]:


columns = df.columns
binary_cols = []

for col in columns:
    if df[col].value_counts().shape[0] == 2:
        binary_cols.append(col)


# In[12]:


binary_cols # categorical features with two classes


# In[13]:


# Categorical features with multiple classes
multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']


# ## Binary categorical features

# In[14]:


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("gender", data=df, ax=axes[0,0])
sns.countplot("SeniorCitizen", data=df, ax=axes[0,1])
sns.countplot("Partner", data=df, ax=axes[0,2])
sns.countplot("Dependents", data=df, ax=axes[1,0])
sns.countplot("PhoneService", data=df, ax=axes[1,1])
sns.countplot("PaperlessBilling", data=df, ax=axes[1,2])


# In[15]:


churn_numeric = {'Yes':1, 'No':0}
df.Churn.replace(churn_numeric, inplace=True)


# In[16]:


df[['gender','Churn']].groupby(['gender']).mean()


# In[17]:


df[['SeniorCitizen','Churn']].groupby(['SeniorCitizen']).mean()


# In[18]:


df[['Partner','Churn']].groupby(['Partner']).mean()


# In[19]:


df[['Dependents','Churn']].groupby(['Dependents']).mean()


# In[20]:


df[['PhoneService','Churn']].groupby(['PhoneService']).mean()


# In[21]:


df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()


# In[22]:


table = pd.pivot_table(df, values='Churn', index=['gender'],
                    columns=['SeniorCitizen'], aggfunc=np.mean)
table


# In[23]:


table = pd.pivot_table(df, values='Churn', index=['Partner'],
                    columns=['Dependents'], aggfunc=np.mean)
table


# ## Other Categorical Features

# ### Internet Service

# In[24]:


sns.countplot("InternetService", data=df)


# In[25]:


df[['InternetService','Churn']].groupby('InternetService').mean()


# In[26]:


df[['InternetService','MonthlyCharges']].groupby('InternetService').mean()


# In[27]:


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("StreamingTV", data=df, ax=axes[0,0])
sns.countplot("StreamingMovies", data=df, ax=axes[0,1])
sns.countplot("OnlineSecurity", data=df, ax=axes[0,2])
sns.countplot("OnlineBackup", data=df, ax=axes[1,0])
sns.countplot("DeviceProtection", data=df, ax=axes[1,1])
sns.countplot("TechSupport", data=df, ax=axes[1,2])


# In[28]:


df[['StreamingTV','Churn']].groupby('StreamingTV').mean()


# In[29]:


df[['StreamingMovies','Churn']].groupby('StreamingMovies').mean()


# In[30]:


df[['OnlineSecurity','Churn']].groupby('OnlineSecurity').mean()


# In[31]:


df[['OnlineBackup','Churn']].groupby('OnlineBackup').mean()


# In[32]:


df[['DeviceProtection','Churn']].groupby('DeviceProtection').mean()


# In[33]:


df[['TechSupport','Churn']].groupby('TechSupport').mean()


# ### Phone service

# In[34]:


df.PhoneService.value_counts()


# In[35]:


df.MultipleLines.value_counts()


# In[36]:


df[['MultipleLines','Churn']].groupby('MultipleLines').mean()


# ### Contract, Payment Method

# In[37]:


plt.figure(figsize=(10,6))
sns.countplot("Contract", data=df)


# In[38]:


df[['Contract','Churn']].groupby('Contract').mean()


# In[39]:


plt.figure(figsize=(10,6))
sns.countplot("PaymentMethod", data=df)


# In[40]:


df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()


# In[41]:


fig, axes = plt.subplots(1,2, figsize=(12, 7))

sns.distplot(df["tenure"], ax=axes[0])
sns.distplot(df["MonthlyCharges"], ax=axes[1])


# In[42]:


df[['tenure','MonthlyCharges','Churn']].groupby('Churn').mean()


# In[43]:


df[['Contract','tenure']].groupby('Contract').mean()


# In[44]:


df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis=1, inplace=True)


# In[45]:


df.head()


# # Data Preprocessing

# In[46]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# In[47]:


cat_features = ['SeniorCitizen', 'Partner', 'Dependents',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
X = pd.get_dummies(df, columns=cat_features, drop_first=True)


# In[49]:


sc = MinMaxScaler()
a = sc.fit_transform(df[['tenure']])
b = sc.fit_transform(df[['MonthlyCharges']])


# In[50]:


X['tenure'] = a
X['MonthlyCharges'] = b


# In[51]:


X.shape


# # Resampling

# In[52]:


sns.countplot('Churn', data=df).set_title('Class Distribution Before Resampling')


# In[53]:


X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]


# In[54]:


print(len(X_no),len(X_yes))


# In[55]:


X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
print(len(X_yes_upsampled))


# In[56]:


X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)


# In[57]:


sns.countplot('Churn', data=X_upsampled).set_title('Class Distribution After Resampling')


# # ML model

# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)
y = X_upsampled['Churn'] #target (dependent variable)


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# ### Ridge Classifier

# In[61]:


from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[62]:


clf_ridge = RidgeClassifier() #create a ridge classifier object
clf_ridge.fit(X_train, y_train) #train the model


# In[63]:


pred = clf_ridge.predict(X_train)  #make predictions on training set


# In[64]:


accuracy_score(y_train, pred) #accuracy on training set


# In[65]:


confusion_matrix(y_train, pred)


# In[66]:


pred_test = clf_ridge.predict(X_test)


# In[67]:


accuracy_score(y_test, pred_test)


# ### Random Forests

# In[68]:


from sklearn.ensemble import RandomForestClassifier


# In[69]:


clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)


# In[70]:


clf_forest.fit(X_train, y_train)


# In[71]:


pred = clf_forest.predict(X_train)


# In[72]:


accuracy_score(y_train, pred)


# In[73]:


confusion_matrix(y_train, pred)


# In[74]:


pred_test = clf_forest.predict(X_test)


# In[75]:


accuracy_score(y_test, pred_test)


# In[76]:


from sklearn.model_selection import GridSearchCV


# In[77]:


parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}
forest = RandomForestClassifier()
clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)


# In[78]:


clf.fit(X, y)


# In[79]:


clf.best_params_


# In[80]:


clf.best_score_


# In[ ]:





# In[ ]:




