import streamlit as st
from components.sidebar import add_custom_sidebar
import pandas as pd
import itertools
import importlib.util
import time
from hgtk.text import decompose, compose
import sys
import os
# main.py 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(main_dir)

# utils 디렉토리 경로 추가
utils_dir = os.path.abspath(os.path.join(current_dir, "../utils"))
sys.path.append(utils_dir)

from utils.All_Musical_Process import Musical_Process
from utils.All_Musical_Process import Recommender
import config

"""기본 틀"""
# 사이드바 추가
add_custom_sidebar()

# CSS 스타일
st.markdown("""
<style>
.stButton > button {
    background-color: transparent;
    border: none;
    color: black;
    font-size: 24px;
}

.sidebar {
    background-color: #f0f0f0;
    padding: 20px;
}

.search-container {
    background-color: #f5f5f5;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
}

.actor-list-item {
    cursor: pointer;
    margin-bottom: 5px;
    padding: 5px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background-color: #f9f9f9;
}
.actor-list-item:hover {
    background-color: #e6e6e6;
}

</style>
""", unsafe_allow_html=True)


# Musical_Process 실행 함수
def run_musical_process():
    try:
        # All_Musical_Process.py 경로 설정
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/All_Musical_Process.py"))
        # 모듈 로드 및 실행
        spec = importlib.util.spec_from_file_location("All_Musical_Process", script_path)
        all_musical_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(all_musical_module)
    except Exception as e:
        st.error(f"All_Musical_Process 실행 중 오류 발생: {e}")


# 로딩 화면 표시 함수
def show_loading_screen():
    # 빈 컨테이너 생성
    placeholder = st.empty()
    
    # 스피너 표시
    with placeholder.container():
        with st.spinner("로딩 중... 잠시만 기다려주세요(처음에는 로딩이 길어질 수 있습니다.)"):
            run_musical_process()
    placeholder.empty()

show_loading_screen()

# 메인 페이지 제목
st.markdown("# 뮤지컬 chat")

# 상단 설명 텍스트
st.markdown("""
<div class="header-text">
    좋아하시는 배우와 장르를 선택하시면<br>
    맞춤형 뮤지컬을 추천해드립니다
</div>
""", unsafe_allow_html=True)

# 배우 데이터 로드
@st.cache_data
def load_actor_list():
    file = f'{config.file_path}/actor_genre_df.json'
    if not os.path.exists(file):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file}")
    
    actor_genre_df = pd.read_json(file, lines=True)
    
    actor_list = actor_genre_df["actor"].tolist()
    return sorted(actor_list)

actor_list = load_actor_list()

st.markdown("## 배우와 장르 선택")

def get_chosung(text):
    """한글 문자열에서 초성만 추출"""
    result = ""
    for char in text:
        if '가' <= char <= '힣':  # 한글 범위 내에서만 분리
            decomp = decompose(char)
            if decomp[0] != '':  # 초성 존재
                result += decomp[0]
        else:
            result += char  # 한글 외에는 그대로 추가
    return result


# 세션 상태 초기화
if "selected_actor" not in st.session_state:
    st.session_state["selected_actor"] = None
if "favorite_actor" not in st.session_state:
    st.session_state["favorite_actor"] = ""
if "filtered_actors" not in st.session_state:
    st.session_state["filtered_actors"] = []

# 배우 입력창
favorite_actor = st.text_input(
    "좋아하는 배우를 입력하세요",
    placeholder="배우 이름 또는 초성을 입력하세요",
    value=st.session_state["favorite_actor"],
    key="favorite_actor_input"
)

# 검색 처리
if favorite_actor != st.session_state["favorite_actor"]:
    st.session_state["favorite_actor"] = favorite_actor

    search_query = st.session_state["favorite_actor"]

    if search_query:
        # 단어와 초성 구분
        if all('가' <= char <= '힣' for char in search_query):  # 완성된 단어 입력 시
            st.session_state["filtered_actors"] = [
                actor for actor in actor_list if actor[:len(search_query)] == search_query
            ]
        else:  # 초성 입력 시
            user_chosung = get_chosung(search_query)
            st.session_state["filtered_actors"] = [
                actor for actor in actor_list
                if user_chosung == get_chosung(actor[:len(search_query)])  # 초성 매칭
            ]

# 검색 결과 동적 표시
if st.session_state["filtered_actors"]:
    st.markdown("### 검색 결과:")
    for actor in st.session_state["filtered_actors"]:
        if st.button(actor, key=f"actor_{actor}"):
            st.session_state["selected_actor"] = actor
            st.session_state["favorite_actor"] = actor  # 입력창 업데이트
            st.session_state["filtered_actors"] = []  # 검색 결과 초기화
            break

# 선택된 배우 출력
if st.session_state["selected_actor"]:
    st.markdown(f"**선택된 배우:** {st.session_state['selected_actor']}")

# 초기화 버튼
if st.button("초기화"):
    st.session_state["selected_actor"] = None
    st.session_state["favorite_actor"] = ""
    st.session_state["filtered_actors"] = []



# 장르처리
base_genres = {
    1: 'Historical',
    2: 'Romance',
    3: 'Drama',
    4: 'Fantasy',
    5: 'Comedy'
}

# 장르 조합 생성
def generate_genre_combinations(base_genres):
    combinations = list(itertools.combinations(base_genres.items(), 2))
    combined_genres = {
        idx + 6: f"{genre1[1]} & {genre2[1]}"
        for idx, (genre1, genre2) in enumerate(combinations)
    }
    return combined_genres

genre_mapping = {**base_genres, **generate_genre_combinations(base_genres)}

# 장르 선택
genre_choice = st.selectbox(
    "좋아하는 장르를 선택하세요",
    options=list(genre_mapping.keys()),
    format_func=lambda x: f"{x}. {genre_mapping[x]}"
)


# 추천 버튼
if st.button("추천받기"):
    if not st.session_state["selected_actor"]:
        st.error("배우를 선택해주세요.")
    else:
        st.markdown("### 추천 결과")
        with st.spinner("추천 결과를 생성하는 중입니다... 잠시만 기다려주세요."):
            try:
                recommender = Recommender()
                recommender.load_data()
                recommender.load_model()
                
                genre_id = genre_choice
                prediction, similar_musicals = recommender.run(genre_id, st.session_state["selected_actor"])

                if similar_musicals is not None and not similar_musicals.empty:
                    st.markdown("### 추천된 뮤지컬 목록")
                    for idx, (_, musical) in enumerate(similar_musicals.iterrows()):
                        col1, col2 = st.columns([1, 2]) 
                        with col1:
                            if pd.notna(musical["poster"]):
                                st.image(
                                    musical["poster"],
                                    caption=musical["title"],
                                    use_container_width=True
                                )
                        with col2:
                            st.markdown(f"""
                            - **[{idx+1}] 제목**: {musical['title']}
                            - **장소**: {musical['place_x']}
                            - **기간**: {musical['start_date_x']} ~ {musical['end_date_x']}
                            - **출연진**: {musical['cast']}
                            - **장르**: {musical['genre']}
                            - **티켓 가격**: {musical['ticket_price']}원
                            """)
                        if idx == 2:
                            break
                else:
                    st.warning("추천 결과가 없습니다.")
            except Exception as e:
                st.error(f"추천 과정에서 오류가 발생했습니다: {str(e)}")