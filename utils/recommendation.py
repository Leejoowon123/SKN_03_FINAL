from model_loading import ModelHandler
from DataLoading import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import json

class Recommender:
    def __init__(self):
        self.file_path = config.file_path
        # model 경로
        self.model_path = f"{config.model_file_path}/{config.model_name}"
        # final_processing.json
        self.data_path = f"{config.file_path}/{config.last_processing_file_name}"
        self.model = None
        self.df = None
        # embedding.json
        self.embedding_file = config.embedding_file
        self.mapping_name = config.mapping_file_name
        self.genre_mapping = {
            1: "드라마/감동",
            2: "코미디/유머",
            3: "액션/스릴러",
            4: "판타지/어드벤처",
            5: "음악중심/주크박스"
        }
        self.genre_combinations = self._generate_genre_combinations()
        self.fm_features = config.fm_features
        self.deep_features = config.deep_features
        self.label_encoder = LabelEncoder()

    """1~5번 장르의 조합 생성"""
    def _generate_genre_combinations(self):
        combinations = {i: genre for i, genre in self.genre_mapping.items()}
        current_idx = 6
        for i in range(1, 6):
            for j in range(i + 1, 6):
                combinations[current_idx] = f"{self.genre_mapping[i]}, {self.genre_mapping[j]}"
                current_idx += 1
        return combinations    

    """데이터 로드"""
    def load_data(self):
        self.final_df = DataLoader.load_json_to_dataframe(self.data_path)
        self.embedding_df = DataLoader.load_json_to_dataframe(self.embedding_file)
        self.df = self.embedding_df.copy()
        
    """모델 로드"""
    def load_model(self):
        self.model = ModelHandler.load_model(self.model_path)
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.") 

    """매핑한 파일 생성 코드"""
    def create_mapping(self):
        # 매핑 데이터프레임 생성
        self.mapping_df = pd.DataFrame()
        self.actor_genre_df = pd.DataFrame()
        self.load_data()
        max_casts = 8
        for i in range(1, max_casts + 1):
            self.mapping_df[f'cast_{i}_name'] = self.final_df['cast'].apply(
                lambda x: x.split(',')[i - 1].strip() if isinstance(x, str) and len(x.split(',')) > i - 1 else np.nan
            )
        
        self.mapping_df['genre'] = self.final_df['genre']

        embedding_columns = [f'cast_{i}' for i in range(1, max_casts + 1)] + ['genre']
        for col in embedding_columns:
            self.mapping_df[col] = self.embedding_df[col]

        self.mapping_df['genre_embedding'] = self.embedding_df['genre']

        # DataLoader.save_to_json(self.mapping_df, self.file_path, self.mapping_name)        
        print(f"Mapping DataFrame saved to {self.mapping_name}")
        self.actor_genre_df = self.create_actor_genre_mapping(self.mapping_df)
        DataLoader.save_to_json(self.actor_genre_df, self.file_path, 'actor_genre_df.json')
        DataLoader.save_to_json(self.mapping_df, self.file_path, self.mapping_name)
        
    
    """배우가 참여한 장르, 임베딩된 값 조사"""
    def create_actor_genre_mapping(self, mapping_df):

        actor_columns = [f"cast_{i}_name" for i in range(1, 9)]
        embedding_columns = [f"cast_{i}" for i in range(1, 9)]

        """배우 데이터 수집"""
        if not set(actor_columns).issubset(mapping_df.columns):
            raise KeyError(f"Expected columns {actor_columns} not found in mapping_df.")
        
        all_actors = mapping_df[actor_columns].stack().reset_index(drop=True).unique()
        
        actor_genre_df = pd.DataFrame(columns=["actor", "genre"])

        """배우와 장르 매핑"""
        for actor in all_actors:
            if actor and actor != "Unknown":
                genres = mapping_df[
                    mapping_df[actor_columns].apply(lambda row: actor in row.values, axis=1)
                ]['genre'].unique()
                
                actor_num = None
                for i, col in enumerate(actor_columns):
                    if not mapping_df[mapping_df[col] == actor].empty:
                        actor_num = mapping_df[mapping_df[col] == actor][embedding_columns[i]].iloc[0]
                        break

            genre_strings = list(map(str, genres))
            actor_genre_df = pd.concat([
                actor_genre_df,
                pd.DataFrame({
                    "actor": [actor],
                    "genre": [",".join(sorted(genre_strings))],
                    "actor_num": [actor_num]
                })
            ], ignore_index=True)

        return actor_genre_df
    
    """유사한 배우 찾기"""
    def find_most_similar_actor(self, input_actor_embedding, candidate_actors):
        candidate_embeddings = self.actor_genre_df[
            self.actor_genre_df['actor'].isin(candidate_actors)
        ]['actor_num'].values

        input_embedding_vector = np.array([input_actor_embedding]).reshape(1, -1)
        candidate_embedding_matrix = np.array(candidate_embeddings).reshape(-1, 1)
        similarities = cosine_similarity(input_embedding_vector, candidate_embedding_matrix).flatten()

        matching_actors = self.actor_genre_df[
            self.actor_genre_df['actor'].isin(candidate_actors)
        ]
        matching_actors['similarity'] = similarities
        sorted_actors = matching_actors.sort_values(by='similarity', ascending=False)

        return sorted_actors

    
    """배우와 장르를 기반으로 추천 로직 실행"""
    def recommend(self, genre_id, actor):
        self.genre_id = genre_id
        self.actor = actor
        """사용자가 입력한 genre_id에 맞는 배우 찾기"""
        genre_str = str(self.genre_id - 1)
        matching_actors = self.actor_genre_df[
            self.actor_genre_df['genre'].str.contains(genre_str, na=False)
        ]

        if matching_actors.empty:
            print(f"입력한 장르 {self.genre_id}를 포함한 배우를 찾을 수 없습니다.")
            return None

        """genre_id를 가진 배우 목록에서 입력한 actor가 있는지 확인"""
        if actor in matching_actors['actor'].values:
            print(f"입력한 배우 {actor}와 장르 {genre_id}가 매칭됩니다.")
            return self.actor, self.genre_id

        """입력한 actor의 임베딩 가져오기"""
        input_actor_row = self.actor_genre_df[self.actor_genre_df['actor'] == actor]
        if input_actor_row.empty:
            print(f"입력한 배우 {self.actor}를 찾을 수 없습니다.")
            return None

        input_actor_embedding = input_actor_row['actor_num'].values[0]

        """코사인 유사도를 기반 가장 유사한 배우 찾기"""
        sorted_candidates = self.find_most_similar_actor(input_actor_embedding, matching_actors['actor'].values)

        """유사도가 높은 배우 중 genre_id가 포함된 배우 찾기"""
        for _, candidate_row in sorted_candidates.iterrows():
            self.candidate_actor = candidate_row['actor']
            self.candidate_genres = candidate_row['genre'].split(',')
            if genre_str in self.candidate_genres:
                print(f"추천 배우: {self.candidate_actor}, 장르: {self.genre_id}")
                return self.candidate_actor, self.genre_id

        """적합한 배우를 찾지 못한 경우"""
        print(f"입력한 장르 {genre_id}와 매칭되는 배우를 찾을 수 없습니다.")
        return None


    """사용자 입력 데이터 -> 모델 입력 데이터로"""
    def preprocess_input(self, actor, genre):
        # 입력된 actor를 actor_genre_df에서 검색하여 actor_num 값으로 변환
        actor_row = self.actor_genre_df[self.actor_genre_df['actor'] == actor]
        if actor_row.empty:
            raise ValueError(f"입력한 배우 '{actor}'를 찾을 수 없습니다.")
        actor_idx = actor_row['actor_num'].iloc[0]
        genre_idx = genre - 1
        return actor_idx, genre_idx
    

    """모델 입력 데이터 생성"""
    def create_model_inputs(self, actor_idx, genre_idx):

        inputs = {f"fm_input_{feature}": np.zeros((1, 1)) for feature in self.fm_features}
        inputs.update({f"deep_input_{feature}": np.zeros((1, 1)) for feature in self.deep_features})

        # FM features
        inputs["fm_input_genre"] = np.array([[genre_idx]])
        inputs["fm_input_cast_1"] = np.array([[actor_idx]])
        for i in range(2, 9):
            inputs[f"fm_input_cast_{i}"] = np.array([[0]])

        # Deep features
        for feature in self.deep_features:
            if feature in ["place_x", "runtime", "musical_license", "period"]:
                inputs[f"deep_input_{feature}"] = np.array([[0.0]])
            elif feature.startswith("day_time_"):
                inputs[f"deep_input_{feature}"] = np.array([[0]])
            else:
                inputs[f"deep_input_{feature}"] = np.array([[0]])
        return inputs

    """추천 및 유사 뮤지컬 찾기"""
    def recommend_and_find_similar(self, actor, genre):

        if self.model is None:
            self.load_model()
        # 1. 모델에 입력 데이터를 전처리
        actor_idx, genre_idx = self.preprocess_input(actor, genre)
        inputs = self.create_model_inputs(actor_idx, genre_idx)
        prediction = self.model.predict(inputs)
        print(f"추천된 뮤지컬 예측 점수: {prediction}")

        # 3. 유사한 뮤지컬 찾기
        similar_musicals = self.find_similar_musicals(inputs)
        return prediction, similar_musicals

    def find_similar_musicals(self, inputs):
        # 훈련 데이터에서 FM 및 Deep Feature 추출
        training_data = self.df[self.fm_features + self.deep_features].fillna(0).to_numpy()

        # 모델 입력 데이터와 유사도 비교
        input_vector = np.hstack([inputs[f"fm_input_{f}"] for f in self.fm_features] +
                                [inputs[f"deep_input_{f}"] for f in self.deep_features]).flatten()

        # 코사인 유사도 계산
        similarity_scores = cosine_similarity(training_data, input_vector.reshape(1, -1)).flatten()

        # 상위 3개 뮤지컬 찾기
        top_3_indices = np.argsort(similarity_scores)[-3:][::-1]

        # 추천된 뮤지컬의 인덱스를 기반으로 final_processing.json 데이터 로드
        final_df = DataLoader.load_json_to_dataframe(self.data_path)  # final_processing.json

        # 추천된 뮤지컬 출력
        recommended_musicals = final_df.iloc[top_3_indices].drop_duplicates(subset='title')

        if recommended_musicals.empty:
            print("추천할 뮤지컬이 없습니다.")
        else:
            print("추천된 뮤지컬:")
            for idx, row in recommended_musicals.iterrows():
                print(f"- {row['title']}")
        # print("유사한 뮤지컬 3개:")
        # for _, row in recommended_musicals.iterrows():
        #     print(f"뮤지컬 제목: {row['title']}")
        #     print(f"시작 날짜: {row['start_date_x']}")
        #     print(f"종료 날짜: {row['end_date_x']}")
        #     print(f"장소: {row['place_x']}")
        #     print(f"출연진: {row['cast']}")
        #     print(f"러닝타임: {row['runtime']}")
        #     print(f"티켓 가격: {row['ticket_price']}")
        #     print(f"장르: {row['genre']}")
        #     print("-" * 40)  # 각 뮤지컬 정보 구분선
        return recommended_musicals
        

    def run(self, genre_id, actor):
        self.load_data()
        self.create_mapping()
        print(f"사용자 입력 배우: {actor}, 입력 장르: {genre_id}")
        actor, genre_id = self.recommend(genre_id, actor)

        if actor and genre_id:
            prediction, similar_musicals = self.recommend_and_find_similar(actor, genre_id)
            print("추천된 뮤지컬과 유사한 뮤지컬:")
            return prediction, similar_musicals
        print("추천을 완료할 수 없습니다.")
        return None
        
if __name__ == "__main__":

    recommender = Recommender()
    recommender.load_data()
    # 장르 매핑 출력
    print("다음은 선택 가능한 장르입니다:")
    for key, value in recommender.genre_combinations.items():
        print(f"{key}: {value}")

    genre_id = int(input("원하는 장르 번호를 입력하세요(1~15): ").strip())
    actor = input("좋아하는 배우를 입력하세요: ")

    prediction, similar_musicals = recommender.run(genre_id, actor)

    if similar_musicals is not None:
        print("추천된 뮤지컬과 유사한 뮤지컬이 출력됩니다.")

