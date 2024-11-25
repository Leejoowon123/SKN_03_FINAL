import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    """특정 컬럼 제거"""
    def remove_columns(self, columns):
        self.df = self.df.drop(columns=columns, errors='ignore')
        return self.df

    """특정 컬럼의 빈값 제거"""
    def remove_empty_values(self, columns):
        # 빈 문자열 또는 공백, null 있는 문자열 -> NaN
        self.df[columns] = self.df[columns].replace([r'^\s*$', 'null'], np.nan, regex=True)
        # NaN 값이 있는 행 제거
        self.df = self.df.dropna(subset=columns, how='any')
        return self.df

    """저장된 임곗값 초과 데이터 필터링"""
    def filter_numeric_values(self, column_name, threshold):
        # 컬럼을 숫자로 변환, 변환 불가 값은 NaN 처리
        self.df[column_name] = pd.to_numeric(self.df[column_name], errors="coerce")
        
        # NaN 값을 제거
        self.df = self.df.dropna(subset=[column_name])
        
        # 임계값 초과 조건 적용
        self.df = self.df[self.df[column_name] <= threshold]
        
        return self.df



    def all_process(self, remove_columns, remove_empty_columns):
        self.df = self.remove_columns(remove_columns)
        self.df = self.remove_empty_values(remove_empty_columns)
        self.df = self.filter_numeric_values("percentage",100)
        return self.df
