#!/usr/bin/env python
# coding: utf-8

# # import required libraries
# 

# In[ ]:


import numpy as np
import pandas as  pd
import seaborn  as sns
import matplotlib.pyplot as plt


# In[ ]:


hundai=pd.read_csv(r"C:\Users\jafar\Downloads\car_age_price.csv")
hundai


# In[ ]:


hundai.tail()


# In[ ]:


hundai.shape


# In[ ]:


hundai.info()


# In[ ]:


hundai.describe()


# In[20]:


hundai[hundai.duplicated(keep='first')]


# In[21]:


hundai.isnull().sum()


# # grouping

# In[24]:


aggregated_hundai=hundai.groupby('Year')['Price'].mean().reset_index()
aggregated_hundai


# # outlier analysis

# In[26]:


fig,axs=plt.subplots(2,figsize=(5,5))
plt1=sns.boxplot(x=hundai['Year'],ax=axs[0],color='g')
plt2=sns.boxplot(x=hundai['Price'],ax=axs[1],color='b')
plt.tight_layout()


# # model selection

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn import model_selection


# In[29]:


X=aggregated_hundai[['Year']]
y=aggregated_hundai['Price']


# # fit the linear model

# In[30]:


model=LinearRegression()


# In[31]:


model.fit(X,y)


# In[32]:


from ipywidgets.widgets.interaction import interactive_output


# In[33]:


predictions=model.predict(X)


# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


mse = mean_squared_error(y, predictions)
mse


# # print coefficient models

# In[37]:


linear_coefficients = model.coef_
linear_coefficients


# In[38]:


print(model.intercept_,model.coef_)


# # predict price in 2022

# In[39]:


year_to_predict=2023


# In[40]:


predicted_price=model.predict(np.array([[year_to_predict]]))


# In[41]:


print(f"Predicted price for {year_to_predict}: {predicted_price[0]:.2f}")


# In[42]:


plt.scatter(X,y,label='Data points')
plt.plot(X,model.predict(X), color='red', label= 'Regression Line')
plt.xlabel('Year')
plt.ylabel('price')
plt.show()


# In[43]:


preds=model.predict(X)


# # LASSO REGRESSION

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# # creating a lasso regression model

# In[46]:


from sklearn.linear_model import Lasso


# In[47]:


alpha=0.1


# In[48]:


lasso_model=Lasso(alpha=alpha)


# In[49]:


lasso_model.fit(X_train,y_train)


# # make predictions

# In[51]:


y_pred=lasso_model.predict(X_test)


# # calculate the mean squared error

# In[52]:


from sklearn.metrics import mean_squared_error


# In[53]:


mse = mean_squared_error(y_test, y_pred)
mse


# In[54]:


# print model coefficient and mse


# In[55]:


print('Lasso Coefficients:', lasso_model.coef_)
print(f'Mean Squared Error: {mse:.2f}')


# In[57]:


#So the MSE of linear model is 466132094.1418216 and the MSE of lasso model is 872380730.45,then The lower the MSE, the better the model's predictions fit the data. so the the linear model is better.


# In[ ]:




