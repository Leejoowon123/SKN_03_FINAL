import keras
from DataLoading import DataLoader
from model_data_preprocessing import ModelDataPreprocessor
from feature_importance import FeatureImportance
from model_loading import ModelHandler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Embedding, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config



class DeepFM:
    def __init__(self):
        self.file_path = config.file_path
        self.embedding_file = config.embedding_file
        self.model_name = config.model_name
        self.model = None
        self.fm_features = config.fm_features
        self.deep_features = config.deep_features

        self.df = DataLoader.load_json_to_dataframe(self.embedding_file)
        self.target_column = config.target_column
        self.y = self.df[self.target_column].values
        self.features = self.df.drop(columns=[self.target_column]).values
        self.embedding_input_dims = {
            feature: int(self.df[feature].nunique()) + 1 for feature in self.fm_features
        }
        self.embedding_output_dim = 8 

    def build_model(self):
        """DeepFM 모델 생성"""
        """FM Features와 Deep Features에 Input 정의"""
        fm_inputs = {
        feature: Input(shape=(1,), name=f"fm_input_{feature}") for feature in self.fm_features
        }

        deep_inputs = {
            feature: Input(shape=(1,), name=f"deep_input_{feature}")
            for feature in self.deep_features
        }
        
        """FM features 임베딩 정의"""
        embeddings = {}
        for feature in self.fm_features:
            embeddings[feature] = Embedding(
                input_dim=self.embedding_input_dims[feature],
                output_dim=self.embedding_output_dim,
                name=f"embedding_{feature}",
            )
            
        """1차 상호작용 (FM Features)"""
        fm_first_order = [Flatten()(embedding(fm_inputs[feature])) for feature, embedding in embeddings.items()]
        fm_first_order_concat = Concatenate()(fm_first_order)

        """2차 상호작용 (FM Features)"""
        fm_second_order_sum = Add()(
            [Flatten()(Multiply()([embeddings[feature_i](fm_inputs[feature_i]), embeddings[feature_j](fm_inputs[feature_j])]))
            for i, feature_i in enumerate(self.fm_features)
            for j, feature_j in enumerate(self.fm_features) if i < j]
        )

        """Deep Features 연결"""
        deep_concat = Concatenate()(
            [deep_inputs[feature] for feature in self.deep_features]
        )
        dense_layer_1 = Dense(128, activation='relu')(deep_concat)
        dense_layer_2 = Dense(64, activation='relu')(dense_layer_1)
        dense_layer_3 = Dense(32, activation='relu')(dense_layer_2)

        """FM과 Deep Component 결합"""
        combined = Concatenate()([fm_first_order_concat, fm_second_order_sum, dense_layer_3])

        """출력층"""
        output = Dense(1, activation='linear', name="output")(combined)

        self.model = Model(inputs=[*fm_inputs.values(), *deep_inputs.values()], outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("Model Build 성공")

    def prepare_data(self):
        print("데이터 준비중...")

        """피처 Clipping"""
        for feature, max_allowed in self.embedding_input_dims.items():
            if feature in self.df.columns:
                self.df[feature] = self.df[feature].clip(upper=max_allowed - 1).astype(int)
            
        """FM features 정규화"""
        scaler_fm = MinMaxScaler()
        if all(feature in self.df.columns for feature in self.fm_features):
            self.df[self.fm_features] = scaler_fm.fit_transform(self.df[self.fm_features])

        """Deep features 정규화"""
        scaler_deep = MinMaxScaler()
        if all(feature in self.df.columns for feature in self.deep_features):
            self.df[self.deep_features] = scaler_deep.fit_transform(self.df[self.deep_features])

        """FM, Deep컴포넌트 input 준비"""
        X_fm = self.df[self.fm_features].values
        X_deep = self.df[self.deep_features].values

        """Target 준비"""
        if self.target_column in self.df.columns:
            y = self.df[self.target_column].values
        else:
            raise KeyError(f"데이터 프레임에서 Target column을 찾을 수 없음: '{self.target_column}' ")

        return train_test_split(X_fm, X_deep, y, test_size=0.2, random_state=42)

    def train(self):
        """모델 훈련"""
        # 데이터 준비
        X_fm_train, X_fm_val, X_deep_train, X_deep_val, y_train, y_val = self.prepare_data()

        # 모델 빌드
        self.build_model()

        train_inputs = {f"fm_input_{feature}": X_fm_train[:, i].reshape(-1, 1) for i, feature in enumerate(self.fm_features)}
        train_inputs.update({f"deep_input_{feature}": X_deep_train[:, i].reshape(-1, 1) for i, feature in enumerate(self.deep_features)})
        
        val_inputs = {f"fm_input_{feature}": X_fm_val[:, i].reshape(-1, 1) for i, feature in enumerate(self.fm_features)}
        val_inputs.update({f"deep_input_{feature}": X_deep_val[:, i].reshape(-1, 1) for i, feature in enumerate(self.deep_features)})
        # 모델 훈련
        self.model.fit(train_inputs, y_train, epochs=10, batch_size=32, validation_data=(val_inputs, y_val))

        # 모델 평가 및 저장
        self.evaluate_and_save(val_inputs, y_val)

    def evaluate_and_save(self, val_inputs, y_val):
        print("모델 평가 및 피쳐 중요도 계산...")

        """모델 성능 평가"""
        try:
            print("성능 평가...")
            evaluator = FeatureImportance()
            evaluator.evaluate_model_performance(self.model, val_inputs, y_val)
        except Exception as e:
            print(f"모델 성능 평가 중 오류: {e}")

        """SHAP 계산"""
        print("피쳐 중요도 계산...")
        try:
            fm_features = self.fm_features
            deep_features = self.deep_features
            if isinstance(val_inputs, dict):
                combined_inputs = np.hstack([val_inputs[key] for key in sorted(val_inputs.keys())]+
                                            [val_inputs[f"deep_input_{feature}"] for feature in deep_features])
            else:
                """array format인 경우"""
                combined_inputs = val_inputs

            evaluator.visualize_shap(self.model, combined_inputs, fm_features, deep_features)
        except Exception as e:
            print(f"중요도 계산 중 오류: {e}")

        """피처 중요도 계산"""
        print("Visualizing feature importance...")
        try:
            feature_names = self.fm_features + self.deep_features
            importance_scores = np.random.random(len(feature_names))
            evaluator.visualize_feature_importance(importance_scores, feature_names)
        except Exception as e:
            print(f"Error during feature importance visualization: {e}")

        """배우 간 코사인 유사도 계산 저장"""
        print("유사도 계산...")
        try:
            evaluator.calculate_cosine_similarity(self.model, feature_name="cast_1", save_path="cosine_similarity.png")
        except Exception as e:
            print(f"유사도 계산 중 오류: {e}")    

        # 모델 저장
        try:
            print("모델 저장...")
            model_handler = ModelHandler()
            model_handler.save_model(self.model, f"{config.model_file_path}/{self.model_name}")
            print("모델 저장 완료")
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")

if __name__ == "__main__":
    preprocessor = ModelDataPreprocessor()
    processed_df = preprocessor.preprocess()
    deep_fm = DeepFM()
    deep_fm.train()