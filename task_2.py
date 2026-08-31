
from google.colab import files
uploaded = files.upload()

import pandas as pd
import seaborn as sns
dataset =pd.read_csv("Titanic-Dataset.csv")
dataset.head(3)

dataset.describe()
dataset.info()
dataset.select_dtypes(include=['float','int'])

dataset.isnull().sum()
dataset.select_dtypes(include=['int','float'])

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Cabin'] = dataset['Cabin'].fillna(dataset['Cabin'].mode()[0])
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
dataset.isnull().sum()

dataset['Age'].mean()
dataset['Age'].mode()
dataset['Age'].median()


dataset['Embarked'].mode()

import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20,15))
plt.show()

dataset.boxplot(column=['Age'])
plt.show()

dataset.info()

for i in dataset.select_dtypes(include=['float','int']):
    dataset.boxplot(column=[i])
    plt.show()

for i in dataset.select_dtypes(include=['float','int']):
    dataset.hist(bins=5, column=[i])
    plt.show()

numeric_df = dataset.select_dtypes(include=['int64', 'float64'])

print(numeric_df.columns)
print(numeric_df.head())

sns.pairplot(dataset[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])
plt.show()



correlation_data = dataset[['Age','Fare']].corr()
print(correlation_data )
correlation_data.plot()
plt.show()

corr = dataset.select_dtypes(include=['int64', 'float64'])
corr.plot()
plt.show()


