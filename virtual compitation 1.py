#!/usr/bin/env python
# coding: utf-8

# Consider the following Python dictionary `data` and Python list `labels`:
# 
# ``` python
# data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
#         'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
#         'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
# 
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# ```
# 
# **1.** Create a DataFrame `df` from this dictionary `data` which has the index `labels`.

# In[39]:


import pandas as pd
import numpy as np


# In[40]:


data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}


# In[41]:


df=pd.DataFrame(data,index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])


# In[42]:


df


# **2.** Display a summary of the basic information about this DataFrame and its data (*hint: there is a single method that can be called on the DataFrame*).

# In[43]:


df.info() #or df.describe()


# **3.** Return the first 3 rows of the DataFrame `df`.

# In[44]:


df_first_3=df.head(3) #or df.iloc[:3]


# In[45]:


df_first_3


# **4.** Display the 'animal' and 'age' columns from the DataFrame `df`

# In[46]:


df[['animal','age']]


# **5.** Display the data in rows `[3, 4, 8]` *and* in columns `['animal', 'age']'

# In[47]:


df.loc[df.index[[3,4,8]]]


# **6.** Select only the rows where the number of visits is greater than 3.

# In[48]:


df[df['visits']>3]


# **7.** Select the rows where the age is missing, i.e. it is `NaN`.

# In[49]:


df[df['age'].isnull()]


# **8.** Select the rows where the animal is a cat *and* the age is less than 3.

# In[50]:


df[(df['animal']=='cat')&(df['age']<3)]


# **9.** Select the rows where the age is between 2 and 4 (inclusive)

# In[51]:


df[df['age'].between(2,4)]


# **10.** Change the age in row 'f' to 1.5.

# In[52]:


df.loc['f','age']=1.5 
print(df)


# **11.** Calculate the sum of all visits in `df` (i.e. the total number of visits).

# In[53]:


df['visits'].sum()


# **12.** Calculate the mean age for each different animal in `df`.

# In[54]:


df.groupby('animal')['age'].mean()


# **13.** Append a new row 'k' to `df` with your choice of values for each column. Then delete that row to return the original DataFrame.

# In[55]:


df.loc['k']=['cat',3.5,2,'no']
print(df.loc['k'])


# **14.** Count the number of each type of animal in `df`.

# In[56]:


#and then deleting the row
df=df.drop('k')


# **15.** Sort `df` first by the values in the 'age' in *decending* order, then by the value in the 'visits' column in *ascending* order (so row `i` should be first, and row `d` should be last).

# In[57]:


df.sort_values(by=['age','visits'],ascending=[False,True])


# **16.** The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be `True` and 'no' should be `False`.

# In[58]:


df['priority']=df['priority'].map({'yes':True,'no':False})
print(df)


# **17.** In the 'animal' column, change the 'snake' entries to 'python'.

# In[59]:


df['animal']=df['animal'].replace('snake','python')
df['animal']


# **18.** Load the ny-flights dataset to Python

# In[62]:


data=pd.read_csv(r"C:\Users\jafar\Downloads\ny-flights.csv")


# In[63]:


data


# In[ ]:





# **19.** Which airline ID is present maximum times in the dataset

# In[64]:


data['airline_id'].max()


# **20.** Draw a plot between dep_delay and arr_delay

# In[67]:


import matplotlib.pyplot as plt


# In[66]:


plt.bar(data['dep_delay'],data['arr_delay'])
plt.xlabel('dep_delay')
plt.ylabel('arr_delay')
plt.title('Bar Plot of dep_delay by arr_delay')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show


# In[ ]:





# In[ ]:




