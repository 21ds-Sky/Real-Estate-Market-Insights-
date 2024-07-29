#!/usr/bin/env python
# coding: utf-8

# # Data Science Regression Project: Predicting Home Prices in Banglore
# #Dataset is downloaded from here: https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# # Data Load: Load banglore home prices into a dataframe

# In[4]:


df1 = pd.read_csv("bengaluru_house_Data.csv")
df1.head()


# In[5]:


df1.shape


# In[6]:


df1.columns


# In[7]:


df1['area_type'].unique()


# In[8]:


df1['area_type'].value_counts()


# In[9]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# In[10]:


#Data Cleaning: Handle NA values


# In[11]:


df2.isnull().sum()


# In[12]:


df2.shape


# In[13]:


df3 = df2.dropna()
df3.isnull().sum()


# In[14]:


df3.shape


# In[17]:


#Feature Engineering
#Add new feature(integer) for bhk (Bedrooms Hall Kitchen)


# In[18]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[21]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[22]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[23]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[24]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# In[25]:


df4.loc[30]


# In[26]:


(2100+2850)/2


# In[28]:


#Feature Engineering
#Add new feature called price per square feet


# In[29]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[30]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[31]:


df5.to_csv("bhp.csv",index=False)


# In[32]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[33]:


location_stats.values.sum()


# In[34]:


len(location_stats[location_stats>10])


# In[35]:


len(location_stats)


# In[36]:


1287
len(location_stats[location_stats<=10])


# In[37]:


#Dimensionality Reduction
#Any location having less than 10 data points should be tagged as "other" location.


# In[38]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[39]:


len(df5.location.unique())


# In[40]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[41]:


df5.head(10)


# In[46]:


#Outlier Removal Using Business Logic
#As a data scientist when you have a conversation with your business manager 
#(who has expertise in real estate), he will tell you that normally square ft per bedroom is 300 
#(i.e. 2 bhk apartment is minimum 600 sqft. 
#If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. 
#We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft


# In[47]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[48]:


df5.shape


# In[49]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[50]:


#Outlier Removal Using Standard Deviation and Mean


# In[51]:


df6.price_per_sqft.describe()


# In[53]:


#Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. 
#We should remove outliers per location using mean and one standard deviation


# In[54]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[55]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[56]:


plot_scatter_chart(df7,"Hebbal")


# In[57]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[58]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[59]:


plot_scatter_chart(df8,"Hebbal")


# In[60]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[61]:


#Outlier Removal Using Bathrooms Feature


# In[62]:


df8.bath.unique()


# In[63]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[64]:


df8[df8.bath>10]


# In[65]:


df8[df8.bath>df8.bhk+2]


# In[66]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[67]:


df9.head(2)


# In[68]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[69]:


#Use One Hot Encoding For Location


# In[70]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[71]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[72]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# In[73]:


# Now Build a Model


# In[74]:


df12.shape


# In[75]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[76]:


X.shape


# In[77]:


y = df12.price
y.head(3)


# In[78]:


len(y)


# In[79]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[80]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[81]:


#Use K Fold cross validation to measure accuracy of our LinearRegression model


# In[82]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[83]:


#We can see that in 5 iterations we get a score above 80% all the time. 
#This is pretty good but we want to test few other algorithms for regression to see if we can get even better score.
#We will use GridSearchCV for this purpose
#Now Find best model using GridSearchCV


# In[ ]:


from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'n_jobs': [None, 1, 2, 4]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse', 'poisson', 'squared_error', 'absolute_error'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

find_best_model_using_gridsearchcv(X, y)

# Replace X and y with your actual data
# best_model_results = find_best_model_using_gridsearchcv(X, y)


# In[ ]:


#Based on above results we can say that LinearRegression gives the best score. Hence we will use that.
#Now Test the model for few properties


# In[ ]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[ ]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[ ]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[ ]:


predict_price('Indira Nagar',1000, 2, 2)


# In[ ]:


predict_price('Indira Nagar',1000, 3, 3)


# In[ ]:


#Export the tested model to a pickle file


# In[ ]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[ ]:


import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

