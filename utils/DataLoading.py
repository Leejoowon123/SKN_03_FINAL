import pandas as pd  # pandas 라이브러리 임포트

"""데이터 로딩 및 저장을 위한 클래스"""
class DataLoader:
    """df -> json"""
    @staticmethod
    def save_to_json(df, file_path, file_name):
        """데이터프레임을 JSON 파일로 저장합니다."""
        full_path = f'{file_path}/{file_name}'
        df.to_json(full_path, orient='records', lines=True, force_ascii=False)

    """JSON -> df"""
    @staticmethod
    def load_json_to_dataframe(file_path):
        df = pd.read_json(file_path, lines=True)  
        return df
    
