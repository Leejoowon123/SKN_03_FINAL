import pickle

class ModelHandler:

    """모델 저장"""
    @staticmethod
    def save_model(model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    """모델 불러오기"""
    @staticmethod
    def load_model(path):    
        with open(path, 'rb') as f:
            return pickle.load(f)