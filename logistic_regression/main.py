from logistic_regressor import LogisticRegressor
import pandas as pd
from collections import OrderedDict

data = OrderedDict(amount_spent = [50, 10, 20, 5, 95, 70, 100, 200, 0] , send_discount = [0, 1, 1, 1, 0, 0, 0, 0, 1])

df=pd.DataFrame.from_dict(data)

X=df['amount_spent'].astype('float').values
y=df['send_discount'].astype('float').values
X = X.reshape(X.shape[0], 1)

clf = LogisticRegressor()
clf.fit(X,y)

print(clf.predict(X))