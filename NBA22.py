#NBA 2021-2022 Single Variable Linear Regression to predict wins
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

#predicts the number of wins based on defensive rebounds taken as input
reb = float(input("How many defensive rebounds per game does your team get? "))
prediction = reg.predict([[reb]])
print("The predicted wins for {} rebounds is ".format(int(reb)) + str(int(round(prediction[0]))))

#plots the scatterplot with the linear model
plt.plot(df[x], reg.predict(df[[x]].values), color = 'blue')
plt.show() 
