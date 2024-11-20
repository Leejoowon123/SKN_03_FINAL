import pandas as pd
import os
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain_community.llms import OpenAI
from config import add_genre_file_name, input_file_name, file_path

class GenreStoryUpdater:
    def __init__(self):
        self.total_charge = 0
        self.cnt = 0
        self.prompt_template = ChatPromptTemplate.from_template("""
        다음은 뮤지컬의 정보입니다:
        제목: {title}
        상영 위치: {place}
        출연진: {cast}
        포스터 URL: {poster}
        
        1. 이 뮤지컬의 장르를 드라마/감동, 코미디/유머, 액션/스릴러, 판타지/어드벤처, 음악중심/주크박스 이 5가지 카테고리 중에 적절한 것을 골라서 적어주세요.
        2. 이 뮤지컬의 줄거리를 50~100자로 요약해서 적어주세요.
        
        아래와 같은 형식으로 답변해주세요:
        장르: <장르>
        줄거리: <줄거리>
        """)
        self.chat_model = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=300,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

    def update_genre_and_story(self, df):
        for idx, row in df.iterrows():
            if pd.notnull(row['genre']) and pd.notnull(row['story']):
                genre = row['genre']
                story = row['story']
            else:
                genre, story = self.get_genre_and_story(row)

            df.at[idx, "genre"] = genre
            df.at[idx, "story"] = story

    def get_genre_and_story(self, row):
        
        if row['genre'] and row['story']:
            return row['genre'], row['story']
        else:
            try:
                prompt = self.prompt_template.format_messages(
                    title=row['title'],
                    place=row['place'],
                    cast=row['cast'],
                    poster=row['poster']
                )
                
                with get_openai_callback() as cb: 
                    response = self.chat_model(prompt) 
                    content = response.content
                    
                    self.total_charge += cb.total_cost
                    self.cnt += 1
                    print(f"{self.cnt}: 호출에 청구된 총 금액(USD): \t${self.total_charge}")

                genre = content.split("장르: ")[1].split("\n")[0].strip()
                story = content.split("줄거리: ")[1].strip()
                
                return genre, story
            except Exception as e:
                print(f"오류 발생: {e}")
                return "", ""

def main():
    # 파일 경로 설정
    add_genre_story_path = f'{file_path}/{add_genre_file_name}'
    processed_performance_details_path = f'{file_path}/{input_file_name}'
    
    # 파일 존재 여부 확인
    if os.path.exists(add_genre_story_path):
        df = pd.read_json(add_genre_story_path)
    else:
        # processed_perfomance_details.json -> 데이터프레임 생성
        df = pd.read_json(processed_performance_details_path)
        df['genre'] = None
        df['story'] = None

    updater = GenreStoryUpdater()
    updater.update_genre_and_story(df)

    # 파일 저장
    df.to_json(add_genre_story_path, orient='records', lines=True, force_ascii=False)

if __name__ == "__main__":
    main()