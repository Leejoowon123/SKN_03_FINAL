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
from sklearn.inspection import permutation_importance

#부정샘플링한 파일 넣어서 데이터프레임으로 만들기

class MusicalRecommender:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.model = None
        self.label_encoders = {}
        self.vocab_sizes = {}
    
    """데이터 로드 및 모델 데이터 전처리"""
    def load_and_preprocess_data(self):

        self.data = pd.read_csv('c:\\Users\\USER\\Desktop\\이주원코드\\SKN_03_FINAL\\Data\\embedding.csv')
        # 원본 데이터 보존을 위한 copy
        self.original_data = self.data.copy()
        
        categorical_features = ['title', 'cast', 'genre']
        
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            self.data[feature] = self.label_encoders[feature].fit_transform(self.data[feature].astype(str))
            self.vocab_sizes[feature] = len(self.label_encoders[feature].classes_)
        
        # 수치형 변수 정규화
        numerical_features = ['percentage', 'ticket_price'] #percentage는 되어야하나 잘 모르겠다!
        for feature in numerical_features:
            mean = self.data[feature].mean()
            std = self.data[feature].std()
            self.data[feature] = (self.data[feature] - mean) / std



    """DeepFM 모델 생성"""
    def create_deepfm_model(self):

        inputs = {}
        embeddings = []
        
        # FM 컴포넌트를 위한 1차 특성
        first_order_features = []
        
        # 범주형 변수를 위한 임베딩 레이어
        categorical_features = ['title', 'actor', 'genre']
        
        for feature in categorical_features:
            input_layer = Input(shape=(1,), name=feature)
            vocab_size = self.vocab_sizes[feature]
            embedding_dim = min(8, (vocab_size + 1))
            
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
        
        
        # FM과 Deep 컴포넌트 결합 (배우-장르 상호작용 포함)
        combined = Concatenate()([
            first_order_sum,
            second_order,
            deep
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
    
    #학습 데이터 준비
    def prepare_training_data(self):
        # 카테고리형 데이터를 처리하여 임베딩 레이어에 적합하게 변환
        categorical_features = ['title', 'cast', 'genre']
        categorical_data = []
        
        for feature in categorical_features:
            categorical_data.append(self.data[feature].values)
        
        # 수치형 변수는 정규화된 값으로 처리
        numerical_features = ['percentage', 'ticket_price']
        numerical_data = self.data[numerical_features].values
        
        # 카테고리형 데이터와 수치형 데이터를 합친 후, 튜플 형태로 반환
        X = {
            'title': categorical_data[0],
            'cast': categorical_data[1],
            'genre': categorical_data[2],
            'numerical': numerical_data
        }
        
        # 목표 변수 y 정의
        y = self.data['target']
        
        # 8:2 비율로 학습 데이터와 테스트 데이터로 분리 (X와 y 모두 나눠야 함)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    

    """모델 학습"""
    def train_model(self):
        # 학습 데이터 준비
        X_train, X_test, y_train, y_test = self.prepare_training_data()  # 학습 데이터를 준비
        
        # 모델 생성
        self.create_deepfm_model()  # 모델 생성 함수
        
        # 모델 학습
        self.model.fit(
            X_train,  # 학습 데이터
            y_train,  # 학습 목표 변수
            batch_size=64,
            epochs=20,
            verbose=1  # 학습 상태 출력
        )
        
        # 모델 평가
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)  # 테스트 데이터로 모델 평가
        print(f"Test Loss: {test_loss}")
        
        return test_loss

"""피쳐 중요도 계산 함수"""
        # feature_importance 함수 수정
    def calculate_feature_importance(self):
        # 학습 데이터 준비
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        # permutation_importance를 통해 피쳐 중요도 계산
        result = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')
        
        # 각 피쳐의 중요도 반환 (피쳐 이름, 중요도 점수, 표준편차)
        feature_importances = {}
        
        categorical_features = ['title', 'cast', 'genre']
        numerical_features = ['percentage', 'ticket_price']
        
        feature_names = categorical_features + numerical_features
        for i, feature_name in enumerate(feature_names):
            feature_importances[feature_name] = {
                'importance': result.importances_mean[i],
                'std': result.importances_std[i]
            }
        
        # 중요도 출력
        print("Feature Importance:")
        for feature_name, importance_data in feature_importances.items():
            print(f"{feature_name} - Importance: {importance_data['importance']:.4f}, Std: {importance_data['std']:.4f}")
        
        return feature_importances


    """뮤지컬 추천"""
    def recommend_musicals(self, user_input):
        # 사용자 입력을 바탕으로 데이터 생성
        user_data = {
            'title': [],  # 모든 타이틀 ID 추가
            'cast': [],   # 해당 유저가 선호할 캐스트 ID
            'genre': [],  # 선호 장르 ID
            'numerical': []  # 수치형 데이터 (예: 예산, 티켓 가격 등)
        }
        
        # 가능한 뮤지컬 ID에 대해 추천 점수 계산
        all_titles = range(self.vocab_sizes['title'])  # 모든 타이틀 ID
        for title_id in all_titles:
            user_data['title'].append(title_id)
            user_data['cast'].append(user_input['cast'])  # 사용자 선호 캐스트
            user_data['genre'].append(user_input['genre'])  # 사용자 선호 장르
            user_data['numerical'].append(user_input['numerical'])  # 사용자 수치형 입력
        
        # 데이터 프레임 형태로 변환
        X_input = {
            'title': np.array(user_data['title']),
            'cast': np.array(user_data['cast']),
            'genre': np.array(user_data['genre']),
            'numerical': np.array(user_data['numerical'])
        }
        
        # 예측 점수 계산
        predictions = self.model.predict(X_input)
        
        # 점수가 가장 높은 상위 4개의 뮤지컬 선택
        top_indices = predictions.argsort()[-4:][::-1]
        top_titles = [self.label_encoders['title'].inverse_transform([idx])[0] for idx in top_indices]
        
        return top_titles
