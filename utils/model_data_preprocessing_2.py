import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from DataLoading import DataLoader
from Data_preprocessing import DataPreprocessor
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import hashlib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class ModelDataPreprocessor_2:
    def __init__(self):
        self.data = f'{config.file_path}/{config.last_processing_file_name}'
        self.df = DataLoader.load_json_to_dataframe(self.data)
        self.processed_data = None
        self.genre_priority = config.genre_priority
        self.save_path = config.file_path
        self.save_name = config.embedding_file_name
        # 장르 조합 매핑 생성
        self.genre_combination_mapping = self.build_genre_combination_mapping()

    """단일 및 두 장르 조합에 대한 매핑 생성"""
    def build_genre_combination_mapping(self):
        combination_mapping = {**self.genre_priority}
        combination_id = 6  
        for g1 in self.genre_priority:
            for g2 in self.genre_priority:
                if g1 != g2:
                    key = tuple(sorted((g1, g2), key=lambda x: self.genre_priority[x]))
                    if key not in combination_mapping:
                        combination_mapping[key] = combination_id
                        combination_id += 1
        return combination_mapping

    """데이터의 단일 행을 처리하고 결과와 매핑되지 않은 기록 반환"""
    def process_row(self):
        processed_rows = []

        for _, row in self.df.iterrows():
            # 제목 매핑
            title_id = self.unique_titles[row['title']]
            genres = row['genre'].split(', ')
            genres = [g.strip() for g in genres[:2]]  # 공백 제거 및 상위 두 장르로 제한

            if len(genres) == 1:
                genre_id = self.genre_combination_mapping.get(genres[0], 0)
            elif len(genres) == 2:
                # 장르를 우선순위에 따라 정렬한 후 조합 매핑 확인
                normalized_genres = tuple(sorted(genres, key=lambda x: self.genre_priority.get(x, 100)))
                genre_id = self.genre_combination_mapping.get(normalized_genres, 0)
                if genre_id == 0:
                    genre_id = self.genre_combination_mapping.get(genres[0], 0)
            else:
                genre_id = 0

            # 배우별로 행 확장
            cast_list = row['cast'].split(', ')
            for cast_member in cast_list:
                processed_rows.append({
                    "cast": cast_member.strip(),
                    "title": title_id,
                    "editor": ', '.join(row['editor'].split(', ')),
                    "genre": genre_id,
                    "day": row.get('day', ''),
                    "time": row.get('time', ''),
                    "percentage": row.get('percentage', 0.0),
                    "musical_license": 1 if row['musical_license'] == 'Y' else 0,
                    "period": (pd.to_datetime(row['end_date_x'], errors='coerce') -
                            pd.to_datetime(row['start_date_x'], errors='coerce')).days,
                    "ticket_price": row['ticket_price'] / self.max_ticket_price,
                    "target": 1
                })

        return pd.DataFrame(processed_rows)
        
    def assign_cast_ids(self, df):
        if 'cast_id' not in df.columns:  # 'cast_id' 컬럼이 없을 경우 생성
            df['cast_id'] = None  # 'cast_id' 컬럼 초기화 (None 값으로)
        
        # 'cast' 컬럼에서 고유 배우 이름 추출 및 ID 부여
        unique_casts = sorted(df['cast'].unique())
        cast_to_id = {cast: idx for idx, cast in enumerate(unique_casts)}

        # 'cast_id' 컬럼에 값 매핑
        df['cast_id'] = df['cast'].map(cast_to_id)

        return df


    """모델에 적합하게 전처리"""
    def preprocess_data(self, df):

        # 1. editor 고유 ID로 매핑
        all_editors = {editor: idx for idx, editor in enumerate(
            set(editor.strip() for editors in df['editor'].str.split(', ') for editor in editors), start=1
        )}
        df['editor_combined_id'] = df['editor'].apply(
            lambda x: int(hashlib.md5(str(sorted([all_editors[editor.strip()] for editor in x.split(', ')])).encode()).hexdigest(), 16) % (10 ** 8)
        )
        # 3. percentage와 period 정규화 및 로그 변환
        scaler = MinMaxScaler()
        df[['percentage']] = scaler.fit_transform(df[['percentage']])
        df['period'] = np.log1p(df['period'])  # 로그 변환
        df['period'] = scaler.fit_transform(df[['period']])

        # 4. day 데이터 전처리
        df['day'] = df['day'].apply(lambda x: [d.strip() for d in ','.join(x).split(',') if d.strip()])

        # 5. time 데이터 전처리
        df['time'] = df['time'].apply(lambda x: ','.join(x).replace(' ', '').split(','))  # 공백 제거 및 리스트 변환

        self.preprocess_day_and_time()
        self.processed_data = df
        

    def preprocess_day_and_time(self):
        # 요일 맵핑
        days = config.day
        day_indices = {day: idx for idx, day in enumerate(days)}

        def expand_day_range(day_list):
            expanded = []
            for entry in day_list:
                if "HOL" in entry:
                    expanded.extend(['토요일', '일요일'])
                elif " ~ " in entry:
                    start_day, end_day = entry.split(" ~ ")
                    if start_day in day_indices and end_day in day_indices:
                        start_idx, end_idx = day_indices[start_day], day_indices[end_day]
                        if start_idx <= end_idx:
                            expanded.extend(days[start_idx:end_idx + 1])
                        else:
                            expanded.extend(days[start_idx:] + days[:end_idx + 1])
                else:
                    expanded.append(entry.strip())
            return list(set(expanded))

        self.processed_data['day_vector'] = self.processed_data['day'].apply(
            lambda x: ''.join(['1' if day in expand_day_range(x) else '0' for day in days])
        )
        scaler = MinMaxScaler()
        # Replace `self.processed_data.log1p` with `np.log1p`
        self.processed_data['day_vector'] = np.log1p(self.processed_data['day_vector'].astype(float))
        self.processed_data['day_vector'] = scaler.fit_transform(self.processed_data[['day_vector']])

            # 시간 범주화 (숫자 0~6)
        def categorize_time(times):
            categories = set()
            for time in times:
                hour, minute = map(int, time.split(':'))
                if 10 <= hour < 13 or (hour == 13 and minute == 0):  # 10:00 ~ 13:00
                    categories.add(0)
                elif 13 <= hour < 16 or (hour == 16 and minute <= 30):  # 13:30 ~ 16:30
                    categories.add(1)
                elif (hour == 16 and minute > 30) or (16 < hour <= 24):  # 16:30 ~ 24:00
                    categories.add(2)

            if categories == {0}:
                return 0
            elif categories == {1}:
                return 1
            elif categories == {2}:
                return 2
            elif categories == {0, 1}:
                return 3
            elif categories == {0, 2}:
                return 4
            elif categories == {1, 2}:
                return 5
            elif categories == {0, 1, 2}:
                return 6
            else:
                return -1

            
        self.processed_data['time_category'] = self.processed_data['time'].apply(
            lambda x: categorize_time(x)
        )
        if 'day' in self.processed_data.columns and 'time' in self.processed_data.columns:
            self.processed_data.drop(columns=['day', 'time'], inplace=True)
   
    """모든 배우와 뮤지컬의 조합을 생성하여 target 값을 설정"""
    def expand_cast_musical_combinations(self,df):
            # 모든 배우와 뮤지컬의 고유 목록 생성
            all_casts = df['cast'].unique()
            all_titles = df['title'].unique()

            # 배우-뮤지컬의 모든 조합 생성
            full_combinations = pd.DataFrame(
                [(cast, title) for cast in all_casts for title in all_titles],
                columns=['cast', 'title']
            )

            # 기존 데이터의 출연 여부를 기준으로 target 설정
            df_with_target = df[['cast', 'title', 'target']].drop_duplicates()
            expanded_df = full_combinations.merge(df_with_target, on=['cast', 'title'], how='left').fillna({'target': 0})

            # 나머지 컬럼 정보 병합 (cast_y 포함 방지)
            additional_data = df.drop(columns=['target']).drop_duplicates(subset=['title'])
            if 'cast_y' in additional_data.columns:
                additional_data = additional_data.drop(columns=['cast_y'])

            self.processed_data = expanded_df.merge(
                additional_data,
                on='title',
                how='left'
            )
            if 'cast_y' in self.processed_data.columns and 'cast_x' in self.processed_data.columns:
                self.processed_data.drop(columns=['cast_y'], inplace=True)
                self.processed_data.rename(columns={'cast_x': 'cast'}, inplace=True)

            self.assign_cast_ids(self.processed_data)    


    """언더샘플링과 SMOTE를 결합하여 데이터를 균형화"""
    def balance_data_with_smote_and_undersampling(self,target_column='target', undersampling_ratio=1, random_state=42):
        """샘플링전: 790560, 
        Target 클래스 비율:
        target
        0    0.995528
        1    0.004472"""
        # 다수 클래스와 소수 클래스 분리
        df_majority = self.processed_data[self.processed_data[target_column] == 0]
        df_minority = self.processed_data[self.processed_data[target_column] == 1]


        # 다수 클래스에서 언더샘플링
        df_majority_sampled = df_majority.sample(
            n=min(len(df_majority), len(df_minority) * undersampling_ratio),
            random_state=random_state
        )

        # 샘플링된 데이터 병합
        balanced_data = pd.concat([df_majority_sampled, df_minority]).sample(frac=1, random_state=random_state)

        # 업데이트된 데이터를 저장
        self.processed_data = balanced_data.reset_index(drop=True)
    
    def reduce_large_values(self, value, divisor=1e6, modulo=10000):
        """큰 값을 divisor로 나누어 줄임"""
        return int((value // divisor) % modulo)
    
    def reduce_data(self):
        # 큰 값 줄이기
        self.processed_data['editor_combined_id'] = self.processed_data['editor_combined_id'].apply(self.reduce_large_values)
        # target을 int로 변환
        self.processed_data['target'] = self.processed_data['target'].astype(int)

        # 소수점 4자리로 제한
        for col in ['day_vector', 'percentage', 'period', 'ticket_price','day_time_interaction','actor_genre_preference', 'actor_sales_influence']:
            self.processed_data[col] = self.processed_data[col].round(4)

    """target이 1인 경우 cast가 단 한 번 등장하는 데이터를 제거"""
    def remove_single_appearance_casts(self, df):

        # cast를 기준으로 그룹화하고 target이 1인 경우의 개수 계산
        cast_target_counts = df[df['target'] == 1].groupby('cast')['target'].count()

        # target이 1인 경우가 1번만 등장하는 cast 식별
        single_appearance_casts = cast_target_counts[cast_target_counts <= 2].index

        # 단일 등장 cast 데이터 제거
        self.processed_data = df[~df['cast'].isin(single_appearance_casts)]      

    def calculate_actor_sales_influence(self):
        actor_importance = self.processed_data.groupby(['title', 'cast_id'])['target'].sum().reset_index()
        actor_importance['importance_ratio'] = actor_importance.groupby('title')['target'].transform(lambda x: x / x.sum())
        
        # 중요도에 해당 공연의 판매액을 곱하여 기여도 계산
        title_sales = self.processed_data.groupby('title')['ticket_price'].sum().to_dict()
        actor_importance['actor_sales_influence'] = actor_importance['importance_ratio'] * actor_importance['title'].map(title_sales)
        
        # 결과 반환
        return actor_importance[['cast_id', 'title', 'actor_sales_influence']]


    def add_custom_features(self):

        self.processed_data['day_time_interaction'] = self.processed_data['day_vector'] * self.processed_data['time_category']
        
        # Add actor_sales_influence
        actor_sales = self.calculate_actor_sales_influence()
        self.processed_data = self.processed_data.merge(actor_sales, on=['cast_id', 'title'], how='left')           
        
        # Add actor genre preference
        genre_preference = self.processed_data.groupby(['cast_id', 'genre'])['target'].mean().reset_index()
        genre_vector = genre_preference.pivot(index='cast_id', columns='genre', values='target').fillna(0)
        self.processed_data['actor_genre_preference'] = self.processed_data['cast_id'].map(genre_vector.mean(axis=1).to_dict())

        scaler = MinMaxScaler()
        self.processed_data['actor_genre_preference'] = np.log1p(self.processed_data['actor_genre_preference'])  # 로그 변환
        self.processed_data['actor_genre_preference'] = scaler.fit_transform(self.processed_data[['actor_genre_preference']])

        self.processed_data['actor_sales_influence'] = np.log1p(self.processed_data['actor_sales_influence'])  # 로그 변환
        self.processed_data['actor_sales_influence'] = scaler.fit_transform(self.processed_data[['actor_sales_influence']])

        self.processed_data['day_time_interaction'] = np.log1p(self.processed_data['day_time_interaction'])  # 로그 변환
        self.processed_data['day_time_interaction'] = scaler.fit_transform(self.processed_data[['day_time_interaction']]) 

        self.processed_data['actor_sales_influence'] = self.processed_data['actor_sales_influence'].apply(lambda x: x if x > 0 else 0.005)
        self.processed_data['actor_genre_preference'] = self.processed_data['actor_genre_preference'].apply(lambda x: x if x > 0 else 0.005)       
        


    """전체 데이터를 처리하고 결과를 저장"""
    def run(self):

        self.unique_titles = {title: idx for idx, title in enumerate(self.df['title'].unique())}
        self.max_ticket_price = self.df['ticket_price'].max()
        self.processed_data = self.process_row()
        self.processed_data = self.assign_cast_ids(self.processed_data)
        self.remove_single_appearance_casts(self.processed_data)
        self.preprocess_data(self.processed_data)
        self.add_custom_features()
        self.expand_cast_musical_combinations(self.processed_data)
        self.balance_data_with_smote_and_undersampling()
        self.reduce_data()
        # 결과 저장
        DataLoader.save_to_json(self.processed_data, self.save_path, f'embedding_2.json')
        return self.processed_data


if __name__ == "__main__":
    # 실행 예제
    processor = ModelDataPreprocessor_2()
    processed_df = processor.run()