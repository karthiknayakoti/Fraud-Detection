#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas  as pd 
import numpy as np

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore
sns.set_style("darkgrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# In[2]:


import warnings
warnings.simplefilter(action='ignore')


# In[6]:


df = pd.read_csv('creditcard.csv')
df.tail(10)


# In[7]:


df = pd.read_csv('creditcard.csv')
df.head(10)


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.isna().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


# let's see the duplicate data

duplicated = df.duplicated() 
df_duplicated = df[duplicated]
df_duplicated


# In[13]:


df.describe()


# In[14]:


plt.pie([(df['Class']==0).sum() , (df['Class']==1).sum()], labels=('Normal' , 'Fraud'), explode= [0,.3], autopct= '%1.1f%%', shadow= True)
plt.axis('equal')
plt.show()


# In[15]:


plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=18)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)


# In[16]:


df.hist(bins=50, figsize=(25, 25) , grid = True)
plt.show()


# In[17]:


sns.displot(data=df, x="Time", kind="kde", hue="Class")

plt.rc('ytick', labelsize=10)
plt.ylabel('Density' , size=15, color='#1e90c9')
plt.xlabel('Time' , size=15, color='#1e90c9')
plt.title('Time with Class' , size = 14 , color='#1e90c9')

plt.show()


# In[18]:


sns.displot(data=df, x="Amount", kind="kde", hue="Class")

plt.rc('ytick', labelsize=10)
plt.ylabel('Density' , size=15, color='#1e90c9')
plt.xlabel('Amount' , size=15, color='#1e90c9')
plt.title('Amount with Class' , size = 14 , color='#1e90c9')

plt.show()


# In[19]:


plt.figure(figsize = (10,5))
plt.xlabel('Amount' , size = 15,color='#1e90c9')
plt.ylabel('Class' , size = 15,color='#1e90c9')

plt.plot(df['Amount'] , df['Class'])
plt.show()


# In[20]:


plt.figure(figsize = (20,10))
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.xlabel('Time' , size = 20,color='#1e90c9')
plt.ylabel('Amount' , size = 20,color='#1e90c9')

sns.scatterplot(data = df , x = 'Time' , y = 'Amount' , hue = 'Class')
plt.show()


# In[23]:


plt.figure(figsize = (20,10))
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.title('2D' , size = 40 , y = 1.05 , color='#1e90c9')

sns.scatterplot(x = X2D[:,0] , y = X2D[:,1] , hue =df['Class'] , alpha = .8 )
plt.show()


# In[24]:


# i will drop duplicate values
df = df.drop_duplicates()
print(Fore.BLUE + f"the number of fraud = {(df['Class']==1).sum()} after i drop duplicate values")


# In[25]:


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers , scaled feature= feature−median / IQR.
rob_scaler = RobustScaler()            

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)


# In[26]:


scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)
# Amount and Time are Scaled!

df.head()


# In[27]:


# tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# models
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# metrics
from sklearn.metrics import precision_score, make_scorer, recall_score, f1_score , confusion_matrix , classification_report , ConfusionMatrixDisplay ,roc_curve ,roc_auc_score , precision_recall_curve , auc


# In[28]:


x = df.drop('Class', axis=1)
y = df['Class']


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True , stratify=y, random_state=42)


# In[30]:


print("Training set anomalies:", sum(y_train == 1))
print("Testing set anomalies:", sum(y_test == 1))


# In[31]:


# This function to plot roc curve and compute area under the curve

def plot_roc_curve(fpr, tpr, label = None):
    
    plt.plot(fpr, tpr, linewidth = 2, label= 'auc= '+ str(label))
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate", fontname = "monospace", fontsize = 15, weight = "semibold")
    plt.ylabel("True Positive Rate(Recall)", fontname = "monospace", fontsize = 15, weight = "semibold")
    plt.title("ROC Curve", fontname = "monospace", fontsize = 17, weight = "bold")
    plt.axis([0, 1, 0, 1])
    plt.legend(loc=4)
    plt.show()


# In[32]:


model_if = IsolationForest(n_estimators = 500, contamination = .02 , random_state=42)
model_if.fit(x_train)


# In[33]:


y_pred = model_if.predict(x_test)

# Convert predictions to binary labels (-1 for outlier, 1 for inlier)
y_pred_binary = [1 if label == -1 else 0 for label in y_pred]


# In[34]:


ConfusionMatrixDisplay.from_predictions(y_test , y_pred_binary)
plt.title("Confusion Matrix", fontname = "monospace", fontsize = 15, weight = "bold")
plt.show()


# In[35]:


print("\nClassification Report:\n")
print(Fore.BLUE + classification_report(y_test, y_pred_binary))
precision, recall, _ = precision_recall_curve(y_test, y_pred_binary)
auprc = auc(recall, precision)
auc = roc_auc_score(y_test , y_pred_binary)
print("auc:", auc)
print("AUPRC:", auprc)


# In[ ]:




