#
#

'data pre for breast-cancer-winsconsin'

__author__ = 'hjkruclion'

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

#read data
col_names = list(map(str, np.arange(1, 12)))
data = pd.read_csv('breast-cancer-wisconsin.data.txt', names=col_names)

#std data, not use '?'
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')

#random divide to train(25%) and test(75%)
x_train, x_test, y_train, y_test = train_test_split(data[col_names[1:10]], data[col_names[10]], test_size=0.25, random_state=1)
print(y_train.value_counts())
print(y_test.value_counts())