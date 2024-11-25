import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


"""데이터 로딩 및 저장을 위한 클래스"""
class DataLoader:
    """df -> json"""
    @staticmethod
    def save_to_json(df, file_path, file_name):
        full_path = f'{file_path}/{file_name}'
        df.to_json(full_path, orient='records', lines=True, force_ascii=False)

    """JSON -> df"""
    @staticmethod
    def load_json_to_dataframe(file_path):
        # if file_path == f"{config.file_path}/{config.input_file_name}":
        #     df = pd.read_json(file_path, lines=False)
        # else:
        df = pd.read_json(file_path, lines=True)
        return df
    
