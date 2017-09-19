#
#

'learn digits'

__author__ = 'hjkruclion'

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

#load data and pre look on it
digits = load_digits()
# print(digits)
# print(digits)
print(type(digits.data))

#pre on data, divide to train and test
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print(x_train.shape)
print(x_test.shape)

#std input data
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

#use svm to predict
lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
lsvc_y_predict = lsvc.predict(x_test)

#show report
print(lsvc.score(x_test, y_test))
print(classification_report(y_test, lsvc_y_predict))
