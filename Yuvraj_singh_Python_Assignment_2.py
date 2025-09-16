import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
''' 1) Generate a NumPy array of 10 numbers and 
     compute their square using broadcasting.'''
# Answer : 
numbers = np.arange(10)
squared_numbers = numbers ** 2
print("squared numbers is :", squared_numbers)

''' 2) Plot linear vs. non-linear relationships using Matplotlib.'''
# Answer : For linear 
x = np.arange(1, 11)
y = 2*x + 3
plt.figure(figsize=(12, 5))
plt.plot(x, y)
plt.show()
# Answer : For Non-Linear
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.figure(figsize=(12, 5))
plt.plot(x, y)
plt.show()
''' 3) Load the Seaborn tips dataset and 
      detect outliers using a boxplot.'''
# Answer :
tips = sns.load_dataset('tips')
sns.boxplot(tips)
plt.show()

''' 4)  Compute mean and median for 
a dataset with an outlier and observe the difference.'''
# Answer :
array = np.array([1, 2, 3, 4, 5, 100])
mean = np.mean(array)
median = np.median(array)
print("Mean is :", mean)
print("Median is: ", median)

''' 5) One-hot encode a categorical column using Pandas.'''
# Answer :
data = ['apple', "samsung", "nokia", "lenovo", "oneplus"]
df = pd.DataFrame(data, columns=['Brand'])
encoded = pd.get_dummies(df, columns=['Brand']) #one hot encoding
encoded = encoded.astype(int)
print(encoded)

''' 6) Train a Linear Regression model 
    on a simple dataset and calculate RMSE.'''
# Answer :
names = ["Yuvraj", "Singh", "Ram", "Noor", "Shaina",]
# study hours
X = np.array([2,4,5,7,9]).reshape(-1, 1)
# scores 
y = np.array([50,60,65,70,85])
df = pd.DataFrame({'Names': names, 'study hours' : X.flatten(), "Scores" : y})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE is : ", rmse)
''' 7) Compare RMSE with and without outliers in tips dataset.'''
# Answer :
# Load the tips dataset 
tips = sns.load_dataset('tips')
#check
print(tips.describe())
tips_with_outliers = tips.copy()
model = LinearRegression()
X = tips_with_outliers[['total_bill']]
y = tips_with_outliers['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_with_outliers = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE with outlier", rmse_with_outliers)
# remove the outliers 
Q1 = tips['tip'].quantile(0.25)
Q3 = tips['tip'].quantile(0.75)
IQR = Q3 - Q1
filtered = tips[(tips["tip"] >= Q1 - 1.5 * IQR) & tips["tip"] <= Q3 + 1.5 * IQR]
X = filtered[['total_bill']]
y = filtered['tip']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse_without_outliers = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE without outlier", rmse_without_outliers)
