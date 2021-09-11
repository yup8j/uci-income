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
# describe一下数值型数据
print(df.describe())


def one_hot_convert(dataframe: pd.DataFrame, df_col):
    """
    One hot 转换器
    :param dataframe: 原始dataframe
    :param df_col: 需要转换的列名
    :return:
    """
    # drop掉原来的列
    df_drop = dataframe.drop(columns=df_col, axis=1)
    # 使用get_dummies方法进行onehot，参考 https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
    df_converted = pd.get_dummies(dataframe[df_col])

    return pd.concat([df_drop, df_converted], axis=1, join='inner') #返回拼接结果


# 原始数据
print(df.head())

# 转换workclass
df = one_hot_convert(df, "workclass")
print(df.head())
