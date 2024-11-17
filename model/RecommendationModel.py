import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
import tensorflow as tf

class MusicalRecommender:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.model = None
        self.label_encoders = {}
        self.vocab_sizes = {}
        
        # 한글 컬럼명과 영문 매핑 딕셔너리
        self.column_mapping = {
            '뮤지컬 제목': 'musical_title',
            '관람일': 'viewing_date',
            '관람요일': 'viewing_day',
            '관람 시간': 'viewing_time',
            '티켓 가격': 'ticket_price',
            '판매액': 'sales_amount',
            '일별 예매율': 'daily_booking_rate',
            '출연진': 'cast',
            '공연 시설명': 'venue',
            '공연장 최대 수용 수': 'max_capacity',
            '줄거리': 'plot',
            '공연 장르': 'genre'
        }
        
        # 배우 컬럼 매핑
        self.actor_columns = [f'actor_{i+1}' for i in range(5)]
    
    """데이터 로드 및 전처리"""
    def load_and_preprocess_data(self):

        self.data = pd.read_csv('Data/Final/Combined_Musical_Data.csv')
        # 원본 데이터 보존을 위한 copy
        self.original_data = self.data.copy()
        # 컬럼명 영문으로
        self.data = self.data.rename(columns=self.column_mapping)
        # 출연진 컬럼 분리
        actor_df = pd.DataFrame(self.data['cast'].str.split(',').tolist(), 
                              columns=self.actor_columns)
        self.data = pd.concat([self.data, actor_df], axis=1)
        
        # 레이블 인코딩
        categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
        
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.data[feature] = self.label_encoders[feature].fit_transform(self.data[feature].astype(str))
            self.vocab_sizes[feature] = len(self.label_encoders[feature].classes_)
        
        # 수치형 변수 정규화
        numerical_features = ['ticket_price', 'max_capacity']
        for feature in numerical_features:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            self.data[feature] = (self.data[feature] - mean) / std

    """DeepFM 모델 생성"""
    def create_deepfm_model(self):

        inputs = {}
        embeddings = []
        
        # 범주형 변수를 위한 임베딩 레이어
        categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
        
        for feature in categorical_features:
            input_layer = Input(shape=(1,), name=feature)
            vocab_size = self.vocab_sizes[feature]
            embedding_dim = min(8, (vocab_size + 1) // 2)
            
            embedding = Embedding(
                vocab_size,
                embedding_dim,
                name=f'embedding_{feature}'
            )(input_layer)
            
            embedding_flat = Flatten(name=f'flatten_{feature}')(embedding)
            inputs[feature] = input_layer
            embeddings.append(embedding_flat)
        
        # 수치형 변수
        numerical_input = Input(shape=(2,), name='numerical')
        inputs['numerical'] = numerical_input
        embeddings.append(numerical_input)
        
        # 특성 조합
        concatenated = Concatenate()(embeddings)
        
        # Deep 컴포넌트
        deep = Dense(64, activation='relu')(concatenated)
        deep = Dense(32, activation='relu')(deep)
        deep = Dense(16, activation='relu')(deep)
        output = Dense(1, activation='sigmoid')(deep)
        
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['mae']
        )
    
    """모델 학습"""
    def train_model(self):
        # 학습 데이터
        X = self.prepare_training_data()
        # 예매율을 0~1 범위로 정규화
        y = self.data['daily_booking_rate'].values / 100  
        
        # 모델 생성
        self.create_deepfm_model()
        
        # 모델 학습
        self.model.fit(
            X,
            y,
            batch_size=32,
            epochs=10,
            verbose=0
        )

    """학습 데이터 준비"""
    def prepare_training_data(self):

        X = {}    
        # 범주형 변수 처리
        categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
        
        for feature in categorical_features:
            if feature in self.actor_columns:
                X[feature] = self.data[feature].values.astype('float32')
            else:
                X[feature] = self.data[feature].values.astype('float32')
        
        # 수치형 변수 처리
        numerical_features = ['ticket_price', 'max_capacity']
        X['numerical'] = self.data[numerical_features].values.astype('float32')
        
        return X

    """예측을 위한 입력 데이터 준비"""
    def prepare_input_data(self, row):

        X = {}
        # 범주형 변수 처리
        categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
        
        for feature in categorical_features:
            if feature in self.actor_columns:
                X[feature] = np.array([row[feature]]).astype('float32')
            else:
                X[feature] = np.array([row[feature]]).astype('float32')
        
        # 수치형 변수 처리
        numerical_features = ['ticket_price', 'max_capacity']
        X['numerical'] = row[numerical_features].values.reshape(1, -1).astype('float32')
        
        return X

    """배우 간 시너지 분석"""
    def analyze_actor_synergy(self, actor1, actor2):

        performances = self.original_data[
            self.original_data['출연진'].apply(lambda x: actor1 in x and actor2 in x)
        ]
        
        if len(performances) == 0:
            return 0
        
        avg_booking_rate = performances['일별 예매율'].mean()
        return avg_booking_rate

    """배우의 판매액 영향 분석"""
    def analyze_actor_sales_impact(self, actor):

        performances = self.original_data[
            self.original_data['출연진'].apply(lambda x: actor in x)
        ]
        
        if len(performances) == 0:
            return 0
        
        avg_sales = performances['판매액'].mean()
        return avg_sales

    """배우의 장르 선호도 분석"""
    def analyze_actor_genre_preference(self, actor):

        performances = self.original_data[
            self.original_data['출연진'].apply(lambda x: actor in x)
        ]
        
        if len(performances) == 0:
            return {}
        
        genre_counts = performances['공연 장르'].value_counts()
        total_performances = len(performances)
        
        genre_preferences = {
            genre: (count / total_performances) * 100 
            for genre, count in genre_counts.items()
        }
        
        return genre_preferences

    """추천"""
    def recommend_musicals(self, favorite_actor, favorite_genre):

        predictions = []
        
        # 배우 분석
        actor_sales_impact = self.analyze_actor_sales_impact(favorite_actor)
        actor_genre_pref = self.analyze_actor_genre_preference(favorite_actor)
        
        for idx, row in self.original_data.iterrows():
            # 출연진 리스트 생성
            actors = row['출연진'].split(',')
            has_favorite_actor = favorite_actor in actors
            
            # 시너지 점수 계산
            synergy_score = 0
            if has_favorite_actor:
                for other_actor in actors:
                    if other_actor != favorite_actor:
                        synergy_score += self.analyze_actor_synergy(favorite_actor, other_actor)
                synergy_score = synergy_score / (len(actors) - 1) if len(actors) > 1 else 0
            
            # 장르 일치 여부 확인
            matches_genre = (row['공연 장르'] == favorite_genre)
            
            # 예측을 위한 데이터 준비
            input_data = self.prepare_input_data(self.data.iloc[idx])
            predicted_rate = self.model.predict(input_data, verbose=0)[0][0]  # 이미 0~1 사이 값
            
            # 가중치 적용 (최대 2배까지만 증가하도록 설정)
            final_score = predicted_rate
            if has_favorite_actor:
                final_score *= 1.2
            if matches_genre:
                final_score *= 1.1
            if synergy_score > 0:
                final_score *= (1 + min(synergy_score, 100) / 200)  # 시너지 점수의 영향력 제한
            
            # 최종 점수를 백분율로 변환
            final_score = min(final_score * 100, 100)  # 100% 넘지 않도록 제한
            
            predictions.append({
                '뮤지컬 제목': row['뮤지컬 제목'],
                '관람일': row['관람일'],
                '관람요일': row['관람요일'],
                '공연 시설명': row['공연 시설명'],
                '공연 장르': row['공연 장르'],
                '티켓 가격': row['티켓 가격'],
                '출연진': row['출연진'],
                '예측 예매율': final_score,
                '시너지 점수': synergy_score
            })
        
        # 예측 예매율이 높은 상위 3개 추천
        top_3_recommendations = sorted(
            predictions,
            key=lambda x: x['예측 예매율'],
            reverse=True
        )[:3]
        
        return top_3_recommendations

    """모델 성능 평가"""
    def evaluate_model(self):
        X = self.prepare_training_data()
        y = self.data['daily_booking_rate'].values / 100
        
        # 모델 평가
        loss, mae = self.model.evaluate(X, y, verbose=0)
        
        # RMSE 계산
        y_pred = self.model.predict(X, verbose=0)
        rmse = np.sqrt(np.mean((y - y_pred.flatten()) ** 2))
        
        return {
            'Loss': loss,
            'MAE': mae,
            'RMSE': rmse,
            'MAE(%)': mae * 100,
            'RMSE(%)': rmse * 100
        }