import pandas as pd
import os
from sklearn import linear_model
import matplotlib.pyplot as plt

#Opening the file into a Dataframe
os.chdir("Desktop")
df = pd.read_csv("NBA22 Per Game Team Stats.csv")

#prints the DataFrame
print(df)

#Creates a scatterplot with Defensive Rebounds on the x-axis and Wins on the y-axis. The x and y can be changed to preference.
x = 'DRB'
y = 'WINS'
plt.xlabel(x)
plt.ylabel(y)
plt.scatter(df[x], df[y], color = 'red', marker = '+')

#Creates a linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[[x]].values, df[[y]].values)

#Predicts the wins for a team with 36 rebounds
print("The predicted wins for 36 rebounds is " + str(reg.predict([[36.0]])))

#plots the scatterplot with the linear model
plt.plot(df[x], reg.predict(df[[x]]), color = 'blue')
plt.show()