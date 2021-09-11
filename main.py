import pandas as pd
import numpy as np

# 读入数据，header=None表示没有列名
df = pd.read_csv('./adult.data', header=None)

# 展示前5行数据
print(df.head())

# 利用pandas分析列性质，object为非数值型数据
print(df.info())

# 我抄的，从adult.names
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income']
# 给列取名

df.columns = column_names

print(df.info())