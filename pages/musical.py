import streamlit as st
from components.sidebar import add_custom_sidebar
from SKN_03_FINAL.model.RecommendationModel import MusicalRecommender
import pandas as pd
import os
import subprocess

"""전처리된 파일의 존재 여부 확인 후 없는 경우만 Preprocessing.py를 실행하는 함수"""
def check_and_run_preprocessing():
    
    preprocessed_file_path = '이따 다시쓰자 지원아..'
    
    if not os.path.exists(preprocessed_file_path):
        st.warning("데이터가 없음. 전처리 시작...")
        try:
            # Preprocessing.py 실행
            subprocess.run(['python', 'model/Preprocessing.py'], check=True)
            
            if not os.path.exists(preprocessed_file_path):
                raise Exception("전처리 과정 실패")
                
            st.success("전처리 완료")
        except Exception as e:
            st.error(f"전처리 중 오류 발생: {str(e)}")
            st.stop()
    return True

# 사이드바 추가
add_custom_sidebar()

# 전처리 확인 프로세스
check_and_run_preprocessing()

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
</style>
""", unsafe_allow_html=True)

# 메인 페이지 제목
st.markdown("# 뮤지컬 chat")

# 상단 설명 텍스트
st.markdown("""
<div class="header-text">
    좋아하시는 배우와 장르를 선택하시면<br>
    맞춤형 뮤지컬을 추천해드립니다
</div>
""", unsafe_allow_html=True)

# 모델 및 데이터 사용
@st.cache_resource
def load_recommender():
    recommender = MusicalRecommender()
    recommender.load_and_preprocess_data()
    recommender.create_deepfm_model()
    recommender.train_model()
    
    # 모델 성능 평가
    metrics = recommender.evaluate_model()
    st.markdown("\n=== 모델 성능 평가 ===")
    st.text(f"Loss: {metrics['Loss']:.4f}")
    st.text(f"MAE: {metrics['MAE']:.2f}%")
    st.text(f"RMSE: {metrics['RMSE']:.2f}%")
    st.text(f"R2 Score: {metrics['R2 Score']:.4f}")
    
    st.markdown("\n=== 피처 중요도 ===")
    for feature, importance in metrics['Feature Importance'].items():
        st.text(f"{feature}: {importance:.2f}%")
    
    st.markdown("\n=== 예측값 통계 ===")
    pred_stats = metrics['Prediction Stats']
    st.text(f"평균: {pred_stats['Mean']:.2f}%")
    st.text(f"표준편차: {pred_stats['Std']:.2f}%")
    st.text(f"최소값: {pred_stats['Min']:.2f}%")
    st.text(f"최대값: {pred_stats['Max']:.2f}%")
    
    return recommender

# 모델 로드
try:
    recommender = load_recommender()
except Exception as e:
    st.error(f"모델 로딩 중 오류 발생: {str(e)}")
    st.stop()

# 장르 매핑
genre_mapping = {
    1: 'Historical',
    2: 'Romance',
    3: 'Drama',
    4: 'Fantasy',
    5: 'Comedy'
}

# 폼
with st.form(key='musical_form'):
    # 배우 입력
    favorite_actor = st.text_input("좋아하는 배우", placeholder="배우 이름을 입력하세요")
    
    # 장르 선택 (숫자로 선택)
    genre_choice = st.selectbox(
        "좋아하는 장르를 선택하세요",
        options=list(range(1, 6)),
        format_func=lambda x: f"{x}. {genre_mapping[x]}"
    )
    
    submit = st.form_submit_button("추천받기")

if submit:
    if not favorite_actor.strip():
        st.error("배우 이름을 입력해주세요.")
    else:
        try:
            favorite_genre = genre_mapping[genre_choice]
            
            st.text("\n추천 뮤지컬을 찾는 중...")
            with st.spinner('추천 뮤지컬을 찾는 중...'):
                recommendations = recommender.recommend_musicals(favorite_actor, favorite_genre)
            
            if recommendations:
                st.markdown("### 챗봇 왈 : 당신에게 추천드리는 뮤지컬입니다.")
                # 추천 결과 출력
                for i, rec in enumerate(recommendations, 1):
                    if i == 1:
                        # 첫 번째 추천작은 더 크게 표시
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.image("static/images/display_image_1.jpg", width=400)
                        with col2:
                            st.markdown(f"""
                            ### {rec['뮤지컬 제목']}
                            - 예측 예매율: {rec['예측 예매율']:.2f}%
                            - 공연일: {rec['관람일']} ({rec['관람요일']})
                            - 장르: {rec['공연 장르']}
                            - 공연장: {rec['공연 시설명']}
                            - 티켓 가격: {rec['티켓 가격']:,}원
                            - 출연진: {rec['출연진']}
                            """)
                    else:
                        # 나머지 추천작
                        st.markdown(f"""
                        #### {i}번째 추천 뮤지컬
                        - 제목: {rec['뮤지컬 제목']}
                        - 예측 예매율: {rec['예측 예매율']:.2f}%
                        - 공연일: {rec['관람일']} ({rec['관람요일']})
                        - 장르: {rec['공연 장르']}
                        - 공연장: {rec['공연 시설명']}
                        - 티켓 가격: {rec['티켓 가격']:,}원
                        - 출연진: {rec['출연진']}
                        """)
            else:
                st.warning(f"\n{favorite_actor}와(과) {favorite_genre} 장르의 공연을 찾을 수 없습니다.")
                
        except Exception as e:
            st.error(f"추천 과정에서 오류가 발생했습니다: {str(e)}")