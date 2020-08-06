#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from pycaret.classification import*


# In[2]:


os.chdir('D:\project\Santandor project')


# In[3]:


df=pd.read_csv('train.csv')


# In[4]:


df['target'].value_counts()


# In[5]:


df_null=df.isnull().sum()
df_null=pd.DataFrame(df_null)
df_null[df_null[0]>0]


# In[6]:


target_0=df[df.target==0]
target_1=df[df.target==1]


# In[7]:


df_1_upsampled=resample(target_1,replace=True,n_samples=62160)
df_new=pd.concat([target_0,df_1_upsampled])


# In[8]:


df_new.target.value_counts()


# In[9]:


x_features=set(df_new.columns)-set(['target','ID_code'])


# In[10]:


y=df_new.target
x=sm.add_constant(df_new[x_features])


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[12]:


logit=sm.Logit(y_train,x_train)
model_1=logit.fit()


# In[13]:


def get_significant_vars(model):
    var_p_values=pd.DataFrame(model.pvalues)
    var_p_values['variable']=var_p_values.index
    var_p_values.columns=['pvalues','variable']
    return list(var_p_values[var_p_values.pvalues<=0.05]['variable'])


# In[14]:


significant_var=get_significant_vars(model_1)


# In[15]:


model_2=sm.Logit(y_train,sm.add_constant(x_train[significant_var])).fit()


# In[16]:


y_pred=pd.DataFrame({'actual':y_test,'predicted_prob':model_2.predict(sm.add_constant(x_test[significant_var]))})


# In[17]:


y_pred['predicted']=np.where(y_pred['predicted_prob']>0.5,1,0)


# In[18]:


def draw_roc(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label='ROC Curve(area=%9.2f)'%auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    return fpr,tpr,thresholds


# In[19]:


fpr,tpr,thresholds=draw_roc(y_pred.actual,y_pred.predicted_prob)


# In[20]:


NB_model=GaussianNB().fit(x_train[x_features],y_train)


# In[21]:


y_pred_nb=pd.DataFrame({'actual':y_train,'nb':NB_model.predict(x_train[x_features])})


# In[23]:


fpr,tpr,thresholds=draw_roc(y_pred_nb['actual'],y_pred_nb['nb'])


# In[24]:


logreg_clf=LogisticRegression()


# In[25]:


ada_clf=AdaBoostClassifier(logreg_clf,n_estimators=50)


# In[26]:


ada_clf.fit(x_train[significant_var],y_train)


# In[27]:


y_pred_aboost=pd.DataFrame({'actual':y_train,'aboost':ada_clf.predict(x_train[significant_var])})


# In[31]:


fpr,tpr,thresholds=draw_roc(y_pred_aboost['actual'],y_pred_aboost['aboost'])


# In[34]:


gboost_clf=GradientBoostingClassifier(n_estimators=50,max_depth=10)
gboost_clf.fit(x_train[x_features],y_train)


# In[36]:


y_pred_gboost=pd.DataFrame({'actual':y_train,'gboost':gboost_clf.predict(x_train[x_features])})


# In[37]:


fpr,tpr,thresholds=draw_roc(y_pred_gboost['actual'],y_pred_gboost['gboost'])


# In[70]:


def draw_cm(actual,predicted):
    cm=metrics.confusion_matrix(actual,predicted,[1,0])
    sns.heatmap(cm,annot=True,fmt='.2f',xticklabels=['purchased','not purchased'],yticklabels=['purchased','not purchased'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale(enable=True,axis='y')
    plt.show()


# In[71]:


cm=draw_cm(y_train,y_pred_gboost.gboost)


# In[72]:


print(metrics.classification_report(y_train,y_pred_gboost.gboost))


# # Model Validation

# In[62]:


y_pred_gboost_test=pd.DataFrame({'actual':y_test,'gboost':gboost_clf.predict(x_test[x_features])})


# In[65]:


cm=draw_cm(y_test,y_pred_gboost_test.gboost)


# In[76]:


fpr,tpr,thresholds=draw_roc(y_pred_gboost_test['actual'],y_pred_gboost_test['gboost'])


# In[ ]:


gboost_clf=GradientBoostingClassifier(n_estimators=50,max_depth=10)
cv_scores=cross_val_score(gboost_clf,x_test[x_features],y_test,cv=5,scoring='roc_auc')


# In[73]:


print(metrics.classification_report(y_test,y_pred_gboost_test.gboost))


# In[74]:


y_pred_gboost_all=pd.DataFrame({'actual':y,'gboost':gboost_clf.predict(x[x_features])})


# In[75]:


cm=draw_cm(y,y_pred_gboost_all.gboost)


# In[49]:


print(metrics.classification_report(y,y_pred_gboost_all.gboost))


# In[50]:


fpr,tpr,thresholds=draw_roc(y_pred_gboost_all['actual'],y_pred_gboost_all['gboost'])


# In[51]:


df_test=pd.read_csv('test.csv')


# In[53]:


x_features_test=set(df_test.columns)-set(['ID_code'])


# In[54]:


y_pred_gboost_test=pd.DataFrame({'target':gboost_clf.predict(df_test[x_features])})


# In[56]:


df_test['target']=y_pred_gboost_test.target


# In[ ]:




