import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
train = pd.read_csv('./adult.data', header=None, na_values='?', names=column_names)
test = pd.read_csv('./adult.test', header=None, skiprows=1, names=column_names)


def clean_data(df: pd.DataFrame):
    dummy_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race',
                      'native-country']
    df.replace(' ?', np.nan, inplace=True)
    df['income'] = df['income'].apply(lambda x: 1 if x == ' >50K' else 0)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == ' Male' else 0)
    for feature in dummy_features:
        df = pd.concat([df, pd.get_dummies(df[feature])], axis=1)
        df.drop(feature, axis=1, inplace=True)
    education = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school",
                 "Assoc-acdm", "Assoc-voc", "Some-college", "Bachelors", "Masters", "Doctorate"]
    education_map = {}
    num = 0
    for e in education:
        education_map[e] = num
        num += 1
    df['education'] = df['education'].str.strip().map(education_map)
    return df


# adult.test does not contain "Holand-Netherlands"
df = pd.concat([train, test], axis=0)
df = clean_data(df)
x = df.drop(['income'], axis=1)
x = preprocessing.scale(x)

y = df['income']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_lr = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy score of Logistic Regression is {}".format(accuracy_score(y_true=y_test, y_pred=y_lr)))


dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_dt = dt.predict(x_test)
print("Accuracy score of Decision Tree Classifier is {}".format(accuracy_score(y_true=y_test, y_pred=y_dt)))


# from xgboost import XGBClassifier
#
# xgb = XGBClassifier()
# xgb.fit(x_train,y_train)
# y_xgb = xgb.predict(x_test)
# print("Accuracy score of XGBClassifier is {}".format(accuracy_score(y_true=y_test, y_pred=y_xgb))) #0.8600969084829045


