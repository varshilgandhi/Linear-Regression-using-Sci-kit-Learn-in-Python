# -*- coding: utf-8 -*-
"""
Created on Wed May  5 04:41:49 2021

@author: abc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

#import data
df = pd.read_csv('cells.csv')
print(df)


######################################################################

#plot that data using scatter 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

df = pd.read_csv('cells.csv')
print(df)

plt.xlabel('time')
plt.ylabel('cells')
plt.scatter(df.time,df.cells,color='red',marker='+')


############################################################

#fit or train the model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

df = pd.read_csv('cells.csv')
print(df)

#plt.xlabel('time')
#plt.ylabel('cells')
#plt.scatter(df.time,df.cells,color='red',marker='+')

# x independent (time)
# y dependent - we are predicting y

x_df = df.drop('cells',axis='columns') 

#x_df = df[['time']]   # double brakect indicate dataframe

y_df = df.cells

reg = linear_model.LinearRegression() #Create an instance of the model
reg.fit(x_df,y_df) #Training the model(fitting a line)

print(reg.score(x_df, y_df))   #Predict the score of how well data is predicted 
                               # If score is 1 then it means data is predicted perfectly 

print("Predicted # Cells........",reg.predict([[2.3]]))  #predict the data

# Y = mx + c

c = reg.intercept_
m = reg.coef_
print("From manual calculation, cells = ",(m*2.3+c)) #If we predict the data using this method
                                                     # Then answer is also same

#predict for one single value of x with respect to y

#step : 1 Reading our files

cells_predict_df = pd.read_csv('cells_predict.csv')
print(cells_predict_df)

#step : 2 Predict our file data

predicted_cells = reg.predict(cells_predict_df)
print(predicted_cells)

#add another column in dataframe

cells_predict_df['cells']=predicted_cells
print(cells_predict_df)

cells_predict_df.to_csv("predicted_cells.csv")  #Add and make new csv file

 #############################################################################
 
              #THANK YOU 















#######################################################################################

#predict for one single value of x with respect to y

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
#step : 1 Reading our files

cells_predict_df = pd.read_csv('cells_predict.csv')
print(cells_predict_df.head())

#step : 2 Predict our file data

reg = linear_model.LinearRegression() #Create an instance of the model

predicted_cells = reg.predict(cells_predict_df)
reg.fit(predicted_cells)
print(predicted_cells)

















