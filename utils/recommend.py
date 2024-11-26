import pandas as pd
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class Recommender:
    def __init__(self):
        self.model = None
        self.data = None

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
        except FileNotFoundError:
            raise FileNotFoundError("데이터 파일을 찾을 수 없음")
        
    def load_reference_data(self):
        """기준 데이터 로드"""
        try:
            self.reference_data = pd.read_json(f"{config.file_path}/{config.add_genre_file_name}", lines=True)
        except FileNotFoundError:
            raise FileNotFoundError("기준 파일을 찾을 수 없습니다.")    

    def recommend(self, genre_id, selected_actor):
        """뮤지컬 추천"""
        if self.model is None or self.data is None or self.reference_data is None:
            raise ValueError("모델, 데이터, 또는 기준 데이터가 로드되지 않았습니다.")

        # 선택된 배우와 장르로 필터링
        filtered_data = self.data[
            (self.data['cast'] == selected_actor) &
            (self.data['genre'] == genre_id)
        ]

        if filtered_data.empty:
            print("Filtered data is empty. Check genre_id and selected_actor.")
            return pd.DataFrame()

        # 모델 예측
        X = filtered_data[['title', 'cast', 'genre']].values
        try:
            predictions = self.model.predict(X)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return pd.DataFrame()

        # 예측 결과 추가
        filtered_data['predicted_score'] = predictions
        filtered_data = filtered_data.sort_values(by='predicted_score', ascending=False)

        # 상위 3개 제목 추출
        top_titles = filtered_data['title'].head(3).tolist()

        # 기준 데이터와 매칭
        matched_recommendations = self.reference_data[self.reference_data['title'].isin(top_titles)]

        if matched_recommendations.empty:
            print("No matching titles found in reference data.")
            return pd.DataFrame()

        return matched_recommendations[['title', 'place', 'start_date', 'end_date', 'cast', 'genre', 'ticket_price']]

if __name__ == "__main__":
    recommender = Recommender()
    recommender.load_model()
    recommender.load_data()
    recommender.load_reference_data()

    # 테스트 입력
    genre_id = 1  # 예: Historical
    selected_actor = "홍우진"
    recommendations = recommender.recommend(genre_id, selected_actor)
    print(recommendations)
