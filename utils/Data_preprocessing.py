import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    """특정 컬럼 제거"""
    def remove_columns(self, columns):
        self.df = self.df.drop(columns=columns)
        return self.df

    """특정 컬럼의 빈값 제거"""
    def remove_empty_values(self, columns):
        # 빈 문자열 또는 공백만 있는 문자열 -> NaN
        self.df[columns] = self.df[columns].replace(r'^\s*$', np.nan, regex=True)
        # NaN 값이 있는 행 제거
        self.df = self.df.dropna(subset=columns, how='any')
        return self.df

    def all_process(self, remove_columns, remove_empty_columns):
        self.df = self.remove_columns(remove_columns)
        self.df = self.remove_empty_values(remove_empty_columns)
        return self.df
