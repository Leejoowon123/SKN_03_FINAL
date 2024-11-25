from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np  
import pandas as pd
from model_loading import ModelHandler
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class FeatureImportance:
    
    """SHAP 분석 및 시각화"""
    def visualize_shap(self, model, X_test, fm_features, deep_features):
        # FM 입력 데이터 준비
        fm_inputs = {
            f"fm_input_{feature}": X_test[:, i].reshape(-1, 1)
            for i, feature in enumerate(fm_features)
        }
        # Deep 입력 데이터 준비
        deep_start_idx = len(fm_features)
        deep_inputs = {
            f"deep_input_{feature}": X_test[:, deep_start_idx + i].reshape(-1, 1)
            for i, feature in enumerate(deep_features)
        }

        # SHAP 모델 예측 함수 정의
        def model_predict(inputs):
            reshaped_inputs = {
                **{key: inputs[:, i].reshape(-1, 1) for i, key in enumerate(fm_inputs.keys())},
                **{key: inputs[:, len(fm_inputs) + i].reshape(-1, 1) for i, key in enumerate(deep_inputs.keys())},
            }
            return model.predict(reshaped_inputs).flatten()

        # FM + Deep 통합 입력 데이터 생성
        combined_inputs = np.hstack(list(fm_inputs.values()) + list(deep_inputs.values()))

        # SHAP 설명자 생성
        explainer = shap.KernelExplainer(model_predict, combined_inputs)

        # SHAP 값 계산
        try:
            shap_values = explainer.shap_values(combined_inputs, nsamples=100)

            # SHAP 요약 플롯 생성 및 저장
            feature_names = fm_features + deep_features
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, combined_inputs, feature_names=feature_names, show=False)
            plt.savefig(f"{config.picture_file_path}/SHAP.png", bbox_inches="tight")
            plt.close()
            print("SHAP 시각화가 성공적으로 저장되었습니다.")
        except Exception as e:
            print(f"SHAP 값 계산 중 오류: {e}")


    """Feature Importance Visualization"""
    def visualize_feature_importance(self, importance_scores, feature_names, save_path="Feature_Importance.png"):
        plt.figure(figsize=(10, 8))
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]

        plt.barh(sorted_features, sorted_scores, color='skyblue')
        plt.xlabel("Importance Score")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.savefig(f"{config.picture_file_path}/{save_path}")
        plt.close()
        print("Feature importance visualization saved.")    

    """모델 성능 평가"""
    def evaluate_model_performance(self, model, X_test, y_test):
    
        predictions = model.predict(X_test)

        """로그 역변환"""
        predictions_original = np.expm1(predictions.flatten())
        y_test_original = np.expm1(y_test)

        """MSE & MAE 계산"""
        mse = mean_squared_error(y_test_original, predictions_original)
        mae = mean_absolute_error(y_test_original, predictions_original)
        print(f"Model MSE: {mse:.2f}")
        print(f"Model MAE: {mae:.2f}")

        """시각화"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_original, predictions_original, alpha=0.6)
        plt.plot(
        [min(y_test_original), max(y_test_original)],
        [min(y_test_original), max(y_test_original)],
        color="red",
        linestyle="--",
        label="Ideal Line",
        )
        plt.xlabel("Actual Ticket Price")
        plt.ylabel("Predicted Ticket Price")
        plt.title("Model Performance: Actual vs Predicted")
        plt.text(
            0.05, 0.95, f"MSE: {mse:.2f}\nMAE: {mae:.2f}",
            transform=plt.gca().transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
        )
        plt.legend()
        plt.savefig(f"{config.picture_file_path}/Performance_plot.png")
        plt.close()
        print("Model 성능 사진 저장")

    """배우의 코사인 유사도 계산"""
    def calculate_cosine_similarity(self, model, feature_name, save_path="cosine_similarity.png"):
        embedding_layer_name = f"embedding_{feature_name}"
        try:
            embedding_layer = model.get_layer(embedding_layer_name)
            weights = embedding_layer.get_weights()[0]
        except ValueError:
            print(f"Embedding layer '{embedding_layer_name}'를 찾을 수 없음")
            return

        """cosine 유사도"""
        cosine_sim = cosine_similarity(weights)
        print(f"Cosine 유사도 매트릭스 for '{feature_name}':\n", cosine_sim)

        """시각화"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cosine_sim, cmap='coolwarm', annot=False, fmt='.2f')
        plt.title(f"Cosine Similarity for '{feature_name}'")
        plt.xlabel("Actors")
        plt.ylabel("Actors")
        plt.savefig(f"{config.picture_file_path}/{save_path}")
        plt.show()
        plt.close()
        print(f"Heatmap saved to {save_path}")