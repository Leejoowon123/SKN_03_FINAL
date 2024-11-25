from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

class ModelAnalyzer:
    
    """최적 옵티마이저 찾기"""
    def find_optimal_optimizer(self, model, X, y):
        params = {'optimizer': ['adam', 'sgd']}
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=3)
        grid.fit(X, y)
        return grid.best_params_

    """배우와 장르 선호도 분석"""
    def analyze_actor_genre_preference(self, data):
        actor_genre = data.explode('cast').groupby(['cast', 'genre']).size().unstack(fill_value=0)
        return actor_genre

    """배우 간 시너지 효과 분석"""
    def analyze_actor_synergy(self, data):
        synergy = data['cast'].apply(lambda x: x.split(', ')).explode().value_counts()
        return synergy

    """모델 가중치 최적화"""
    def optimize_weights(self, model, X, y):
        def loss_fn(weights):
            predictions = model.predict(X * weights)
            mse = mean_squared_error(y, predictions)
            return mse

        initial_weights = [1.0] * X.shape[1]
        result = minimize(loss_fn, initial_weights, method='L-BFGS-B')
        optimized_weights = result.x
        print(f"최적화된 가중치: {optimized_weights}")
        return optimized_weights