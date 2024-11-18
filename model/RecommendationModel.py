import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Add, Lambda, Dropout
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from itertools import product

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
        
        # 시간 관련 특성(휴일)
        self.data['is_weekend'] = self.data['viewing_day'].apply(
            lambda x: self.is_weekend(x)
        ).astype(int)
        
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


    """휴일 여부를 판단하는 함수"""
    # 정적 메서드로 선언
    # 클래스 메서드가 아닌 정적 메서드로 선언하여 인스턴스 생성 없이 호출 가능 -> 독립적인 유틸리티 함수처럼 작용
    # 메모리 사용 효율 증가 및 코드 간결성 향상
    @staticmethod
    def is_weekend(viewing_day):
        # 주말(토,일) -> True
        weekend_days = ['Sat', 'Sun']
        return viewing_day in weekend_days
    

    """DeepFM 모델 생성"""
    def create_deepfm_model(self):

        inputs = {}
        embeddings = []
        
        # FM 컴포넌트를 위한 1차 특성
        first_order_features = []
        
        # 범주형 변수를 위한 임베딩 레이어
        categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
        
        for feature in categorical_features:
            input_layer = Input(shape=(1,), name=feature)
            vocab_size = self.vocab_sizes[feature]
            embedding_dim = min(8, (vocab_size + 1) // 2)
            
            # FM 1차 특성
            first_order = Dense(1)(input_layer)
            first_order_features.append(first_order)
            
            # FM 2차 특성과 Deep 컴포넌트를 위한 임베딩
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
        numerical_dense = Dense(1)(numerical_input)
        first_order_features.append(numerical_dense)
        inputs['numerical'] = numerical_input
        embeddings.append(numerical_input)

        # 이진 변수(휴일)
        binary_input = Input(shape=(1,), name='binary')
        binary_dense = Dense(1)(binary_input)
        first_order_features.append(binary_dense)   
        inputs['binary'] = binary_input
        embeddings.append(binary_input)

        # FM 컴포넌트
        # 1차 특성 합
        first_order_sum = Add()(first_order_features)
        
        # 2차 특성 상호작용
        concatenated_embeddings = Concatenate()(embeddings)
        summed_squares = Lambda(lambda x: K.sum(K.square(x), axis=1, keepdims=True))(concatenated_embeddings)
        squared_sum = Lambda(lambda x: K.square(K.sum(x, axis=1, keepdims=True)))(concatenated_embeddings)
        second_order = Lambda(lambda x: 0.5 * (x[0] - x[1]))([squared_sum, summed_squares])
        
        # Deep 컴포넌트
        deep = Dense(256, activation='relu')(concatenated_embeddings)
        deep = Dropout(0.3)(deep)
        deep = Dense(128, activation='relu')(deep)
        deep = Dropout(0.3)(deep)
        deep = Dense(64, activation='relu')(deep)
        deep = Dropout(0.2)(deep)
        
        # 배우-장르 상호작용 레이어
        # 배우와 장르의 임베딩만 추출
        actor_embeddings = []
        genre_embedding = None
        
        for feature in categorical_features:
            if feature in self.actor_columns:
                actor_embeddings.append(inputs[feature])
            elif feature == 'genre':
                genre_embedding = inputs[feature]
        
        # 배우들과 장르 간의 상호작용
        actor_genre_interactions = []
        for actor_input in actor_embeddings:
            interaction = Concatenate()([actor_input, genre_embedding])
            interaction = Dense(32, activation='relu')(interaction)
            actor_genre_interactions.append(interaction)
        
        # 모든 배우-장르 상호작용 결합
        actor_genre_combined = Concatenate()(actor_genre_interactions)
        actor_genre_interaction = Dense(64, activation='relu')(actor_genre_combined)
        actor_genre_interaction = Dropout(0.2)(actor_genre_interaction)
        actor_genre_interaction = Dense(32, activation='relu')(actor_genre_interaction)
        
        # FM과 Deep 컴포넌트 결합 (배우-장르 상호작용 포함)
        combined = Concatenate()([
            first_order_sum,
            second_order,
            deep,
            actor_genre_interaction
        ])
        
        output = Dense(1, activation='sigmoid')(combined)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        # AdamW 옵티마이저 설정
        optimizer = AdamW(
            learning_rate=0.001,
            weight_decay=0.004  # L2 정규화
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
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
            batch_size=64,
            epochs=20,
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
        
        # 수치형 변수 처리 (시간 관련 특성 포함)
        numerical_features = ['ticket_price', 'max_capacity']
        binary_features = ['is_weekend']
        X['numerical'] = self.data[numerical_features].values.astype('float32')
        X['binary'] = self.data[binary_features].values.astype('float32')
        
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
        
        # 수치형 변수 처리 (시간 관련 특성 포함)
        numerical_features = ['ticket_price', 'max_capacity']
        binary_features = ['is_weekend']
        X['numerical'] = row[numerical_features].values.reshape(1, -1).astype('float32')
        X['binary'] = row[binary_features].values.reshape(1, -1).astype('float32')
        return X

    """피처 중요도 계산"""
    def calculate_feature_importance(self):
            X = self.prepare_training_data()
            base_prediction = self.model.predict(X, verbose=0)
            
            feature_importance = {}
            
            # 범주형 변수 중요도
            categorical_features = ['musical_title', 'viewing_day', 'venue', 'genre'] + self.actor_columns
            for feature in categorical_features:
                # 원본 값 저장
                original_values = X[feature].copy()
                # 해당 피처를 섞어서 중요도 측정
                np.random.shuffle(X[feature])
                shuffled_prediction = self.model.predict(X, verbose=0)
                # 중요도 = 원본 예측과 섞은 후 예측의 차이
                importance = np.mean(np.abs(base_prediction - shuffled_prediction))
                feature_importance[feature] = importance
                # 원본 값 복구
                X[feature] = original_values
            
            # 수치형 변수 중요도
            numerical_features = ['ticket_price', 'max_capacity']
            original_values = X['numerical'].copy()
            for i, feature in enumerate(numerical_features):
                temp_values = X['numerical'].copy()
                np.random.shuffle(temp_values[:, i])
                X['numerical'] = temp_values
                shuffled_prediction = self.model.predict(X, verbose=0)
                importance = np.mean(np.abs(base_prediction - shuffled_prediction))
                feature_importance[feature] = importance
            X['numerical'] = original_values
            
            # 이진 변수 중요도
            original_binary = X['binary'].copy()
            np.random.shuffle(X['binary'])
            shuffled_prediction = self.model.predict(X, verbose=0)
            importance = np.mean(np.abs(base_prediction - shuffled_prediction))
            feature_importance['is_weekend'] = importance
            
            # 중요도 정규화
            total_importance = sum(feature_importance.values())
            normalized_importance = {k: v/total_importance * 100 for k, v in feature_importance.items()}
            
            return dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))


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
        # 최적 가중치 계산 
        self.favorite_actors = [favorite_actor]
        self.favorite_genre = favorite_genre
        optimal_weights = self.optimize_weights()
        
        predictions = []
        
        # 배우 분석
        actor_sales_impact = self.analyze_actor_sales_impact(favorite_actor)
        actor_genre_pref = self.analyze_actor_genre_preference(favorite_actor)
        
        # 배우의 평균 판매액 정규화 (0~1 사이)
        max_sales = self.original_data['판매액'].max()
        normalized_sales_impact = min(actor_sales_impact / max_sales, 1.0)  # 1.0으로 제한
        
        # 배우의 장르 선호도 확인
        genre_match_score = actor_genre_pref.get(favorite_genre, 0) / 100
        
        for idx, row in self.original_data.iterrows():
            # 출연진 리스트 생성
            actors = row['출연진'].split(',')
            has_favorite_actor = favorite_actor in actors
            
            # 시너지 점수 계산 (0~1 사이로 정규화)
            synergy_score = 0
            if has_favorite_actor:
                for other_actor in actors:
                    if other_actor != favorite_actor:
                        synergy_score += self.analyze_actor_synergy(favorite_actor, other_actor)
                synergy_score = min(synergy_score / (len(actors) - 1) if len(actors) > 1 else 0, 1.0)
            
            # 장르 일치 여부 확인
            matches_genre = (row['공연 장르'] == favorite_genre)
            
            # 기본 점수 계산 및 정규화
            input_data = self.prepare_input_data(self.data.iloc[idx])
            base_score = self.model.predict(input_data, verbose=0)[0][0]
            
            # 가중치 적용
            weights_sum = 1.0  # 기본 가중치
            if has_favorite_actor:
                weights_sum += optimal_weights['actor_weight']
            if matches_genre:
                weights_sum += optimal_weights['genre_weight']
            if has_favorite_actor:
                weights_sum += optimal_weights['synergy_weight'] * synergy_score
            
            # 정규화된 최종 점수 계산
            final_score = (base_score * weights_sum) * 100
            final_score = min(max(final_score, 0), 100)  # 0~100 범위로 제한
            
            # 선호도 점수 계산
            preference_score = 0
            if has_favorite_actor:
                preference_score += 40  # 배우 매치
            if matches_genre:
                preference_score += 30  # 장르 매치
            if has_favorite_actor and matches_genre:
                preference_score += 30  # 배우-장르 시너지
            
            predictions.append({
                '뮤지컬 제목': row['뮤지컬 제목'],
                '관람일': row['관람일'],
                '관람요일': row['관람요일'],
                '공연 시설명': row['공연 시설명'],
                '공연 장르': row['공연 장르'],
                '티켓 가격': row['티켓 가격'],
                '출연진': row['출연진'],
                '예측 예매율': final_score,
                '선호도 점수': preference_score,
                '시너지 점수': synergy_score * 100,
                '배우 영향력': normalized_sales_impact * 100,
                '장르 선호도': genre_match_score * 100
            })
        
        # 정렬 기준: 선호도 점수를 우선, 그 다음 예측 예매율
        top_3_recommendations = sorted(
            predictions,
            key=lambda x: (x['선호도 점수'], x['예측 예매율']),
            reverse=True
        )[:3]
        
        return top_3_recommendations

    """모델 성능 평가"""
    def evaluate_model(self):
        X = self.prepare_training_data()
        y = self.data['daily_booking_rate'].values / 100
        
        # 모델 평가
        loss, mae, mse = self.model.evaluate(X, y, verbose=0)
        
        # 예측
        y_pred = self.model.predict(X, verbose=0)
        
        # 다양한 메트릭 계산
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        
        # 피처 중요도 계산
        feature_importance = self.calculate_feature_importance()
        
        # 예측값 분포 분석
        pred_mean = np.mean(y_pred) * 100
        pred_std = np.std(y_pred) * 100
        pred_min = np.min(y_pred) * 100
        pred_max = np.max(y_pred) * 100
        
        return {
            'Loss': loss,
            'MAE': mae * 100,
            'RMSE': rmse * 100,
            'R2 Score': r2,
            'Feature Importance': feature_importance,
            'Prediction Stats': {
                'Mean': pred_mean,
                'Std': pred_std,
                'Min': pred_min,
                'Max': pred_max
            }
        }

    """가중치 최적화"""
    def optimize_weights(self):
        # 검증 데이터셋 생성
        X = self.prepare_training_data()
        y = self.data['daily_booking_rate'].values / 100
        X_train, X_val, y_train, y_val = train_test_split(
            self.data, y, test_size=0.2, random_state=42
        )
        
        # 탐색할 가중치 범위
        actor_weights = np.arange(0.2, 0.7, 0.1)  # 0.2 ~ 0.6
        genre_weights = np.arange(0.2, 0.6, 0.1)  # 0.2 ~ 0.5
        synergy_weights = np.arange(0.1, 0.5, 0.1) # 0.1 ~ 0.4
        
        best_weights = {
            'actor_weight': 0.4,
            'genre_weight': 0.3,
            'synergy_weight': 0.3
        }
        best_score = float('-inf')
        
        # Grid Search
        for aw, gw, sw in product(actor_weights, genre_weights, synergy_weights):
            # 가중치 합이 1보다 크지 않도록
            if aw + gw + sw > 1.0:
                continue
            
            val_scores = []
            for idx, row in X_val.iterrows():
                # 기본 예측
                input_data = self.prepare_input_data(row)
                base_score = self.model.predict(input_data, verbose=0)[0][0]
                
                # 가중치 적용
                actors = row['cast'].split(',')
                has_favorite_actor = any(
                    actor in self.favorite_actors for actor in actors
                )
                matches_genre = (
                    row['genre'] == self.label_encoders['genre'].transform([self.favorite_genre])[0]
                )
                
                # 시너지 점수 계산
                synergy_score = 0
                if has_favorite_actor:
                    for actor in actors:
                        if actor in self.favorite_actors:
                            for other_actor in actors:
                                if other_actor != actor:
                                    synergy_score += self.analyze_actor_synergy(actor, other_actor)
                    synergy_score = min(synergy_score / (len(actors) - 1) if len(actors) > 1 else 0, 1.0)
                
                # 가중치 적용
                weights_sum = 1.0
                if has_favorite_actor:
                    weights_sum += aw
                if matches_genre:
                    weights_sum += gw
                if has_favorite_actor:
                    weights_sum += sw * synergy_score
                
                final_score = base_score * weights_sum
                val_scores.append(final_score)
            
            # 검증 세트에 대한 성능 평가
            val_rmse = np.sqrt(np.mean((y_val - val_scores) ** 2))
            val_r2 = r2_score(y_val, val_scores)
            
            # 복합 점수 계산 (RMSE는 낮을수록, R2는 높을수록 좋음)
            current_score = -val_rmse + val_r2
            
            if current_score > best_score:
                best_score = current_score
                best_weights = {
                    'actor_weight': aw,
                    'genre_weight': gw,
                    'synergy_weight': sw
                }
        
        return best_weights