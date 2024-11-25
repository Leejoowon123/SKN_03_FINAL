import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from Data_preprocessing import DataPreprocessor
from model_analyze import ModelAnalyzer
from DataLoading import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class ModelDataPreprocessor:
    def __init__(self):
        self.embedding_file_name = config.embedding_file_name
        self.file_path = config.file_path
        self.last_processing_file_name = config.last_processing_file_name
        self.df = None
        self.analyzer = ModelAnalyzer()

    """runtime을 분 단위 변환"""
    def convert_runtime_to_time(self):
        self.df['runtime'] = self.df['runtime'].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else 0
        )
        return self
    
    """musical_license -> 2진 값으로 변환"""
    def convert_musical_license(self):
        self.df['musical_license'] = self.df['musical_license'].map({'Y': 1, 'N': 0})
        return self

    """start_date와 end_date 차이로 period 컬럼 생성"""
    def add_period_column(self):
        self.df['start_date_x'] = pd.to_datetime(self.df['start_date_x'], errors='coerce')
        self.df['end_date_x'] = pd.to_datetime(self.df['end_date_x'], errors='coerce')
        self.df['period'] = (self.df['end_date_x'] - self.df['start_date_x']).dt.days
        self.df = self.df.drop(columns=['start_date_x', 'end_date_x'])
        return self

    """정규화"""
    def normalize_features(self):
        scaler = MinMaxScaler()
        numeric_cols = ['runtime', 'period', 'ticket_price', 'prfdtcnt', 'seatcnt', 'tickets', 'percentage']
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self
    
    
    """컬럼 별 임베딩 및 추가 피처 생성"""
    def embed_and_process(self):
        """place 임베딩"""
        le = LabelEncoder()
        self.df['place_x'] = le.fit_transform(self.df['place_x'])

        """cast 임베딩"""
        max_casts = self.df['cast'].apply(lambda x: len(x.split(','))).max()
        for i in range(max_casts):
            self.df[f'cast_{i+1}'] = self.df['cast'].apply(
                lambda x: x.split(',')[i].strip() if len(x.split(',')) > i else 'Unknown'
            )
            self.df[f'cast_{i+1}'] = le.fit_transform(self.df[f'cast_{i+1}'])

        """editor 임베딩"""
        max_editors = self.df['editor'].apply(lambda x: len(x.split(','))).max()
        for i in range(max_editors):
            self.df[f'editor_{i+1}'] = self.df['editor'].apply(
                lambda x: x.split(',')[i].strip() if len(x.split(',')) > i else 'Unknown'
            )
            self.df[f'editor_{i+1}'] = le.fit_transform(self.df[f'editor_{i+1}'])

        """genre 카테고리화 및 복합 장르 처리"""
        genre_map = {
            '드라마/감동': 1,
            '코미디/유머': 2,
            '액션/스릴러': 3,
            '판타지/어드벤처': 4,
            '음악중심/주크박스': 5
        }

        def process_genre(genre_str):
            """2개 이상의 장르가 있는 경우 처리"""
            genres = [g.strip() for g in genre_str.split(',')]
            mapped_genres = [genre_map.get(g, None) for g in genres if g in genre_map]
            if len(mapped_genres) == 1:
                return str(mapped_genres[0])  # 단일 장르
            elif len(mapped_genres) > 1:
                return ','.join(map(str, sorted(mapped_genres[:2])))  # 최대 2개의 장르 선택
            return 'Unknown'

        self.df['genre'] = self.df['genre'].apply(process_genre)
        self.df['genre'] = le.fit_transform(self.df['genre'].astype(str))

        """story 키워드 기반 임베딩"""
        self.df['story'] = self.df['story'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

        """day와 time 매핑 후 임베딩"""
        def map_day_and_time(row):
            days = [day.strip() for day in row['day']]
            times = [time.strip() for time in row['time']]
            return [f"{day}-{time}" for day, time in zip(days, times)]
        
        self.df['day_time'] = self.df.apply(map_day_and_time, axis=1)
        max_day_time = self.df['day_time'].apply(len).max()
        for i in range(max_day_time):
            self.df[f'day_time_{i+1}'] = self.df['day_time'].apply(
                lambda x: x[i] if len(x) > i else 'Unknown'
            )
            self.df[f'day_time_{i+1}'] = le.fit_transform(self.df[f'day_time_{i+1}'])

        return self

    """전체 전처리 실행"""
    def preprocess(self):
        
        self.df = DataLoader().load_json_to_dataframe(f'{self.file_path}/{self.last_processing_file_name}')
        self.convert_runtime_to_time()
        self.convert_musical_license()
        self.add_period_column()
        self.normalize_features()
        self.embed_and_process()

        """로그 변환 및 컬럼 제거"""
        self.df['ticket_price'] = self.df['ticket_price'].apply(lambda x: np.log1p(x))
        self.df = DataPreprocessor(self.df).remove_columns(['cast', 'editor', 'age', 'producer', 'host', 'title', 'poster', 'performance_status', 'day_time', 'day', 'time'])
        DataLoader().save_to_json(self.df, self.file_path, self.embedding_file_name)
        return self.df