#use model LogisticRegression to get line divide breast-cancer-NegativeOrPositive
#

'learn tools'

__author__ = 'hjkruclion'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

np.random.seed(2)

#read csv and divid test negative and positive
train = pd.read_csv('breast-cancer-train.csv')
test = pd.read_csv('breast-cancer-test.csv')
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

#look test data
plt.scatter(test_negative['Clump Thickness'].values, test_negative['Cell Size'].values, marker='o', c='red')
plt.scatter(test_positive['Clump Thickness'].values, test_positive['Cell Size'].values, marker='x', c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

#random init line and show
intercept = np.random.random()
coef = np.random.random([2])
lx = np.arange(0, 12)
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='yellow')
plt.scatter(test_negative['Clump Thickness'].values, test_negative['Cell Size'].values, marker='o', c='red')
plt.scatter(test_positive['Clump Thickness'].values, test_positive['Cell Size'].values, marker='x', c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()

#liner logisticRegression get line
lr = LogisticRegression()
lr.fit(train[['Clump Thickness', 'Cell Size']].values, train['Type'].values)
intercept = lr.intercept_
coef = lr.coef_[0]
ly = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly, c='yellow')
plt.scatter(test_negative['Clump Thickness'].values, test_negative['Cell Size'].values, marker='o', c='red')
plt.scatter(test_positive['Clump Thickness'].values, test_positive['Cell Size'].values, marker='x', c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()








