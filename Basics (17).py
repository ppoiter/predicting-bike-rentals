
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
get_ipython().magic('matplotlib inline')
bike_rentals = pd.read_csv('bike_rental_hour.csv')
bike_rentals.head(10)


# In[2]:


plt.hist(bike_rentals['cnt'], 20) 


# In[3]:


corrmat = bike_rentals.corr()
cnt_corr = bike_rentals.corr()['cnt']


# In[4]:


cnt_corr


# In[5]:


sns.heatmap(corrmat)


# In[6]:


def assign_label(hour):
    if hour >= 0 and hour < 6:
        return 4
    elif hour >= 6 and hour < 12:
        return 1
    elif hour >= 12 and hour < 18:
        return 2
    elif hour >= 18 and hour < 24:
        return 3
    else:
        return ('NaN')


# In[7]:


hourly_rentals = bike_rentals['hr']
bike_rentals['time_label'] = bike_rentals['hr'].apply(assign_label)
bike_rentals.head(30)


# Use MSE as error metric, performs well on continuous data, matches our data best

# In[8]:


cutoff = bike_rentals.shape[0] * 0.8
train = bike_rentals.sample(frac = 0.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]


# In[9]:


bike_rentals.columns


# In[10]:


columns = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
        'time_label']

# other candidates to remove: yr


# In[11]:


bike_rentals['yr'].value_counts()


# Actually also remove year, values split about halfway

# In[12]:


columns = ['season', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
        'time_label']


# Since weekday and working day dont seem to have any correlation with cnt, remove these too

# In[13]:


columns = ['season', 'mnth', 'hr', 'holiday', 'weathersit', 'temp',
           'atemp', 'hum', 'windspeed', 'time_label']


# In[14]:


lr = LinearRegression()
lr.fit(train[columns], train['cnt'])
predictions = lr.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# My error higher than the demonstration, who excluded less columns. Try adding a few back in

# In[15]:


columns = list(train.columns)
columns.remove("cnt")
columns.remove("casual")
columns.remove("registered")
columns.remove("dteday")


# In[16]:


lr.fit(train[columns], train["cnt"])
predictions = lr.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# error still slightly higher than theirs, but still lower than without other columns. The lesson here is dont automatically remove predictors which dont seem to be correlated to target in any way. The other lesson is that the error is still high, probably caused by days with lots of hiring

# In[17]:


tree = DecisionTreeRegressor(min_samples_leaf = 5)


# In[18]:


tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# My error is still slightly higher than test but much lower. Experiment with some extra min_samples_leaf values

# In[20]:


tree = DecisionTreeRegressor(min_samples_leaf = 2)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[21]:


tree = DecisionTreeRegressor(min_samples_leaf = 10)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[22]:


tree = DecisionTreeRegressor(min_samples_leaf = 50)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[23]:


tree = DecisionTreeRegressor(min_samples_leaf = 20)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[24]:


tree = DecisionTreeRegressor(min_samples_leaf = 15)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[25]:


tree = DecisionTreeRegressor(min_samples_leaf = 13)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[26]:


tree = DecisionTreeRegressor(min_samples_leaf = 7)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[27]:


tree = DecisionTreeRegressor(min_samples_leaf = 9)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[28]:


tree = DecisionTreeRegressor(min_samples_leaf = 11)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[29]:


tree = DecisionTreeRegressor(min_samples_leaf = 10)
tree.fit(train[columns], train['cnt'])
predictions = tree.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# 7 looks optimal but always remember random state matters. Min samples leaf is the minimum allowable samples to be a leaf node

# In[31]:


#Random Forest, tweak min samples and n_estimators


# In[32]:


forest = RandomForestRegressor(min_samples_leaf = 7, n_estimators = 5)
forest.fit(train[columns], train['cnt'])
predictions = forest.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[33]:


forest = RandomForestRegressor(min_samples_leaf = 7, n_estimators = 10)
forest.fit(train[columns], train['cnt'])
predictions = forest.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[34]:


forest = RandomForestRegressor(min_samples_leaf = 7, n_estimators = 30)
forest.fit(train[columns], train['cnt'])
predictions = forest.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# Worth noting here that error improves all the time but also runs noticeably slower, even on the relatively small dataset.

# In[35]:


forest = RandomForestRegressor(min_samples_leaf = 7, n_estimators = 2)
forest.fit(train[columns], train['cnt'])
predictions = forest.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# In[37]:


forest = RandomForestRegressor(min_samples_leaf = 7, n_estimators = 5)
forest.fit(train[columns], train['cnt'])
predictions = forest.predict(test[columns])
mse = mean_squared_error(predictions, test['cnt'])
mse


# Random Forest accuracy better and overfitting avoided
