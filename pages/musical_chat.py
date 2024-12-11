from dotenv import load_dotenv
import os

load_dotenv()

from datetime import datetime
import streamlit as st
from uuid import uuid4
from components.agent_module import run_agent
import sys
from components.agent_module import run_agent, cast_list
# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(main_dir)

utils_dir = os.path.abspath(os.path.join(current_dir, "../utils"))
sys.path.append(utils_dir)

import config
from utils.recommend import Recommender
from components.tool_module import tools
# from langgraph.prebuilt.tool_executor import ToolExecutor

# # ToolExecutor 생성 (Deprecation 경고 있음)
# tool_executor = ToolExecutor(tools)
from langgraph.prebuilt.tool_node import ToolNode

tool_executor = ToolNode(tools)



# Recommender 초기화
recommender = Recommender()
recommender.load_model()
recommender.load_data()
recommender.load_reference_data()

# 디버깅 출력
print(f"[DEBUG] Actors in cast_list: {cast_list}")
print(f"[DEBUG] Genres in config.unique_genres: {config.unique_genres}")


# Streamlit 세션 초기화
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit 사이드바 설정
st.sidebar.title("뮤지컬 추천 가이드")
st.sidebar.write("채팅방 목록:")

for session_id, session_data in st.session_state.chat_sessions.items():
    if st.sidebar.button(session_data["title"], key=session_id):
        st.session_state.current_session_id = session_id
        st.session_state.chat_history = session_data["messages"]

if st.sidebar.button("새 채팅방 시작"):
    new_session_id = str(uuid4())
    st.session_state.chat_sessions[new_session_id] = {
        "title": f"채팅방 {len(st.session_state.chat_sessions) + 1}",
        "messages": [],
        "created_at": datetime.now(),
    }
    st.session_state.current_session_id = new_session_id
    st.session_state.chat_history = []

# 현재 선택된 채팅방 가져오기
current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id)

# 메인 화면 설정
st.title("뮤지컬 챗봇")
st.markdown("w")
# 채팅방이 선택된 경우
def new_func(agent_input):
    agent_response = run_agent(agent_input)
    extracted_actor = agent_response.get("actor", None)
    extracted_genre = agent_response.get("genre", None)
    return extracted_actor,extracted_genre

if current_session:
    st.write(f"**{current_session['title']}**")

    # 저장된 메시지 출력
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 받기
    prompt = st.chat_input("뮤지컬 관련 질문을 입력해주세요 ~!")

    if prompt:
        # 사용자 메시지 저장
        user_message = {"role": "user", "content": prompt, "timestamp": datetime.now()}
        st.session_state.chat_history.append(user_message)
        current_session["messages"].append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        # LLM 에이전트 호출
        agent_input = {
            "input": prompt,
            "chat_history": st.session_state.chat_history,
            "intermediate_steps": [],
            "agent_scratchpad": "",
            "genres": ", ".join(config.unique_genres),
            "actors": ", ".join(cast_list),
        }
                # 디버깅 출력
        print(f"[DEBUG] Agent input: {agent_input}")

        agent_outcome = run_agent(agent_input)

        # 디버깅 출력
        print(f"[DEBUG] Agent response: {agent_outcome}")

        # 배우와 장르 유효성 검증
        extracted_actor = agent_outcome.get("actor", None)
        extracted_genre = agent_outcome.get("genre", None)

        try:
            extracted_actor, extracted_genre = new_func(agent_input)

            st.markdown(
                f"추천 요청: 배우 - {extracted_actor}, 장르 - {extracted_genre}"
            )

            # 추천 로직 호출
            if extracted_actor and extracted_genre:
                recommendations = recommender.recommend(extracted_actor, extracted_genre)
                if not recommendations.empty:
                    with st.chat_message("assistant"):
                        st.markdown("추천된 뮤지컬:")
                    for _, row in recommendations.iterrows():
                        st.markdown(
                            f"{row['poster']} 
                            - **{row['title']}** (장소: 
                            {row['place']}, 
                            배우: {row['cast']}, 
                            장르: {row['genre']}, 
                            가격: {row['ticket_price']})"
                        )
                else:
                    with st.chat_message("assistant"):
                        st.markdown("죄송합니다. 해당 조건에 맞는 추천이 없습니다.")
            else:
                with st.chat_message("assistant"):
                    st.markdown("추천을 위해 배우와 장르 정보를 확인할 수 없었습니다.")
        except Exception as e:
            st.error(f"에이전트 결과를 처리하는 중 오류가 발생했습니다: {str(e)}")
else:
    st.write("안녕하세요 ☺️ 뮤지컬 관람 계획 중이신가요? \n\n왼쪽에서 채팅방을 선택하거나 새로 시작하세요.")
