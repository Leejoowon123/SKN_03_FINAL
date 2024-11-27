import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class Recommender:
    def __init__(self):
        self.model = None
        self.data = None
        self.reference_data = None
        self.label_encoders = {}

    def load_model(self):
        """모델 로드"""
        try:
            with open(config.save_model_path, 'rb') as file:
                self.model = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("저장된 모델을 찾을 수 없음")

    def load_data(self):
        """데이터 로드"""
        try:
            self.data = pd.read_json(config.df_with_negatives_path, lines=True)
            # 레이블 인코더 로드
            for column in ['title', 'cast', 'genre']:
                self.label_encoders[column] = {val: idx for idx, val in enumerate(self.data[column].unique())}
        except FileNotFoundError:
            raise FileNotFoundError("데이터 파일을 찾을 수 없음")
        
    def load_reference_data(self):
        """기준 데이터 로드"""
        try:
            self.reference_data = pd.read_json(f"{config.file_path}/{config.add_genre_file_name}", lines=True)
        except FileNotFoundError:
            raise FileNotFoundError("기준 파일을 찾을 수 없습니다.")    

    def recommend(self, cast, genre):
        """뮤지컬 추천"""
        if self.model is None or self.data is None or self.reference_data is None:
            raise ValueError("모델, 데이터, 또는 기준 데이터가 로드되지 않았습니다.")

        # 입력값 레이블 인코딩
        if cast not in self.label_encoders['cast']:
            print(f"입력된 cast '{cast}'가 데이터에 존재하지 않습니다.")
            return pd.DataFrame()
        if genre not in self.label_encoders['genre']:
            print(f"입력된 genre '{genre}'가 데이터에 존재하지 않습니다.")
            return pd.DataFrame()
        
        print(f"Debug: 입력된 cast - {cast}")
        print(f"Debug: 입력된 genre - {genre}")

        cast_encoded = self.label_encoders['cast'][cast]
        genre_encoded = self.label_encoders['genre'][genre]
        title_candidates = None
        # 타이틀 후보군 생성
        title_candidates = list(self.label_encoders['title'].values())
        X = pd.DataFrame({
            'title': title_candidates,
            'cast': [cast_encoded] * len(title_candidates),
            'genre': [genre_encoded] * len(title_candidates)
        })

        # 모델 예측
        try:
            predictions = self.model.predict([X['title'], X['cast'], X['genre']])
        except Exception as e:
            raise RuntimeError(f"모델 예측 중 오류 발생: {e}")
        
        # 상위 10개 타이틀 추출
        X['predicted_score'] = predictions
        top_titles = X.sort_values(by='predicted_score', ascending=False).head(10)['title'].tolist()

        # 타이틀 디코딩
        decoded_titles = [k for k, v in self.label_encoders['title'].items() if v in top_titles]
        matched_recommendations = None
        # 기준 데이터와 매칭
        matched_recommendations = self.reference_data[self.reference_data['title'].isin(decoded_titles)]

        if matched_recommendations.empty:
            print("No matching titles found in reference data.")
            return pd.DataFrame()

        return matched_recommendations[['poster','title', 'place', 'start_date', 'end_date', 'cast', 'genre', 'ticket_price']]

if __name__ == "__main__":
    recommender = Recommender()
    recommender.load_model()
    recommender.load_data()
    recommender.load_reference_data()

    """테스트 입력"""
    # cast = "김주호"
    # genre = "음악중심/주크박스"
    # recommendations = recommender.recommend(cast, genre)
    # print(recommendations)
