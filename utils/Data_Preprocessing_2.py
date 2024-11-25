import pandas as pd
import re
from utils.DataLoading import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class Data_Preprocessor:
    def __init__(self, file_path, file_name):
        self.file_path = f"{file_path}"
        self.df = DataLoader.load_json_to_dataframe(f'{self.file_path}/{config.input_file_name}')
        self.last_processing_file_name = config.last_processing_file_name
        file_path = config.file_path
        

    def run(self):
        self.df = self.preprocess()
        self.df.to_json(f'{self.file_path}/{self.last_processing_file_name}', orient='records', lines=True, force_ascii=False)
        print(f'저장 완료')

    def preprocess(self):
        """1. cast와 editor: '등' 부분을 제거"""
        self.df['cast'] = self.df['cast'].str.replace('등$', '', regex=True)
        self.df['editor'] = self.df['editor'].str.replace('등$', '', regex=True)

        """2. runtime: 데이터 형식 변환"""
        self.df['runtime'] = self.df['runtime'].apply(self.convert_runtime)

        """3. ticket_price: 데이터 전처리 함수"""
        self.df['ticket_price'] = self.df['ticket_price'].apply(self.process_ticket_price)

        """4. time: 데이터 전처리 함수"""
        self.df[['day', 'time']] = self.df['time'].apply(self.process_time_column).apply(pd.Series)

        return self.df

    def convert_runtime(self, runtime):
        try:
            if not runtime or not isinstance(runtime, str):
                raise ValueError(f"유효하지 않은 runtime: {runtime}")

            # 공백 제거 및 변환 준비
            runtime = runtime.strip().replace('시간', ':').replace('분', '').replace(' ', '')

            if ':' in runtime:
                hours, minutes = runtime.split(':')
                hours = int(hours) if hours.isdigit() else 0
                minutes = int(minutes) if minutes.isdigit() else 0
            elif runtime.isdigit():
                total_minutes = int(runtime)
                hours = total_minutes // 60
                minutes = total_minutes % 60
            else:
                raise ValueError(f"알 수 없는 runtime 형식: {runtime}")

            # 시간과 분을 반환
            return f"{hours}:{minutes:02d}"
        except Exception as e:
            print(f"runtime 변환 오류: {runtime}, 오류 메시지: {e}")
            return "0:00"

    def process_time_column(self, time_column):
        try:
            if not time_column or '(' not in time_column:
                return [], []

            # 시간 정보를 분리하여 리스트화
            day_time_pairs = re.findall(r"([^()]+)\(([^)]+)\)", time_column)

            # day와 time 각각 추출
            days = [pair[0].strip() for pair in day_time_pairs]
            times = [pair[1].strip() for pair in day_time_pairs]

            return days, times
        except Exception as e:
            print(f"time 변환 오류: {time_column}, 오류 메시지: {e}")
            return [], []

    def process_ticket_price(self, ticket_price):
        try:
            # 1. None이나 "무료" 처리
            if ticket_price is None or '무료' in ticket_price:
                return 0

            # 2. "전석" 포함 여부 확인
            if '전석' in ticket_price:
                if '무료' in ticket_price:
                    return 0
                # "전석" 뒤의 숫자 추출
                price_match = re.search(r'전석.*?(\d+)', ticket_price.replace(',', ''))
                return int(price_match.group(1)) if price_match else 0

            # 3. "석" 이후 ~ "원" 전까지 숫자 추출
            prices = [
                int(price)
                for price in re.findall(r'석.*?(\d+)', ticket_price.replace(',', ''))
            ]

            # 4. 짝수 개수일 경우 중앙 두 값의 평균 반환
            if len(prices) % 2 == 0:
                mid_left = len(prices) // 2 - 1
                mid_right = len(prices) // 2
                return (prices[mid_left] + prices[mid_right]) // 2

            # 5. 홀수 개수일 경우 중앙값 반환
            mid = len(prices) // 2
            return prices[mid]
        except Exception as e:
            print(f"ticket_price 변환 오류: {ticket_price}, 오류 메시지: {e}")
            return 0

def main():    
    Data_Preprocessor.run()

if __name__ == "__main__":
    main()