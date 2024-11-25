import os
from DataLoading import DataLoader
from Data_preprocessing import DataPreprocessor
from prompt import main as prompt_main
from Data_Preprocessing_2 import Data_Preprocessor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


"""데이터 전처리"""
class Processing:
    def __init__(self):
        self.input_file_name = config.input_file_name
        self.file_path = config.file_path
        self.output_file_name = config.output_file_name
        self.columns_to_remove = config.columns_to_remove
        self.columns_with_empty_values = config.columns_with_empty_values
        self.add_genre_file_name = config.add_genre_file_name
        self.last_processing_file_name = config.last_processing_file_name
        self.df = None


    def run(self):
        # """processed_performance_details.json 유무 확인"""
        # if not os.path.exists(f'{self.file_path}/{self.output_file_name}'):
        #     self.df = DataLoader.load_json_to_dataframe(f'{self.file_path}/{self.input_file_name}')

        #     # 2. 데이터 전처리
        #     preprocessor = DataPreprocessor(self.df)
        #     print("Before preprocessing:", self.df.shape)
        #     self.df = preprocessor.all_process(self.columns_to_remove, self.columns_with_empty_values)
        #     print("After preprocessing:", self.df.shape)

        #     # 3. 데이터 저장
        #     DataLoader.save_to_json(self.df, self.file_path, self.output_file_name)
        #     print(f"Processed data saved to {self.file_path}/{self.output_file_name}")

        #     # 4. GenreStoryUpdater 실행
        #     # prompt_main()
        # else:
        #     print(f"{self.output_file_name} already exists. Skipping processing.")

        # """final_processing.json 유무 확인 """
        # if not os.path.exists(f'{self.file_path}/{self.last_processing_file_name}'):
        #     # 5. Data_Processor 실행
        #     preprocessor_2 = Data_Preprocessor(self.file_path, self.add_genre_file_name)
        #     self.df = preprocessor_2.run()
        # else:
        #   print(f'{self.last_processing_file_name} already exists. Skipping processing.')
        
        # 임시방편
        df = DataLoader.load_json_to_dataframe("C:/SKN_3_MyProject/SKN_03_FINAL/Data/Final/merged_output.json")
        preprocessor = DataPreprocessor(df)
        pre_df = preprocessor.all_process(self.columns_to_remove, self.columns_with_empty_values)
        DataLoader.save_to_json(pre_df, self.file_path, self.last_processing_file_name)



if __name__ == "__main__":
    """전처리 실행"""
    main_process = Processing()
    main_process.run()