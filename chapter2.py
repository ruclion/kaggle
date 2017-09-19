#
#

'data pre for breast-cancer-winsconsin'

__author__ = 'hjkruclion'

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


#read data
col_names = list(map(str, np.arange(1, 12)))
data = pd.read_csv('breast-cancer-wisconsin.data.txt', names=col_names)

#std data, not use '?'
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

#random divide to train(25%) and test(75%)
x_train, x_test, y_train, y_test = train_test_split(data[col_names[1:10]], data[col_names[10]], test_size=0.25, random_state=2)
print(y_train.value_counts())
print(y_test.value_counts())

#pre std
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#logistic fit
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

#SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)

#LinerSVC
lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
lsvc_y_predict = lsvc.predict(x_test)

print(lr_y_predict, sgdc_y_predict)

#report
print(classification_report(y_test, lr_y_predict))
print(classification_report(y_test, sgdc_y_predict))
print(classification_report(y_test, lsvc_y_predict))
