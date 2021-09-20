import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

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


test = clean_data(test)
train = clean_data(train)

