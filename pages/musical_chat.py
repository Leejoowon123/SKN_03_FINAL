from dotenv import load_dotenv
import os

load_dotenv()

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
from langgraph.prebuilt.tool_node import ToolNode
from datetime import datetime
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import traceback

tool_executor = ToolNode(tools)

file_path = f"{config.file_path}/{config.add_genre_file_name}"
add_genre_data = pd.read_json(file_path, lines=True)


recommender = Recommender()
recommender.load_model()
recommender.load_data()
recommender.load_reference_data()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_actor" not in st.session_state:
    st.session_state.last_actor = None
if "last_genre" not in st.session_state:
    st.session_state.last_genre = None
if "recommendations" not in st.session_state:
    st.session_state["recommendations"] = None
if "active_titles" not in st.session_state:
    st.session_state["active_titles"] = []
if "recommend_images" not in st.session_state:
    st.session_state["recommend_imgaes"] = []

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

current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id)

st.title("뮤지컬 챗봇")
st.markdown("좋아하는 배우를 입력해주세요")
st.markdown("### 사용 가능한 장르 목록")
st.markdown(", ".join(config.unique_genres))

def new_func(agent_input):
    agent_response = run_agent(agent_input)
    extracted_actor = agent_response.get("actor", None)
    extracted_genre = agent_response.get("genre", None)
    return extracted_actor,extracted_genre

def fetch_interpark_ticket_url(keyword):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # 브라우저 창 안 보이도록 설정
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    base_url = f"https://tickets.interpark.com/contents/search?keyword={keyword}&start=0&rows=20"

    try:
        driver.get(base_url)
        wait = WebDriverWait(driver, 10)
        element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/main/div/div/div[1]/div[2]/a")))
        data_prd_no = element.get_attribute("data-prd-no")

        if data_prd_no:
            final_url = f"https://tickets.interpark.com/goods/{data_prd_no}"
            return final_url
        else:
            return None

    except Exception as e:
        print(f"Error fetching Interpark ticket URL: {e}")
        return None
    finally:
        driver.quit()


if current_session:
    st.write(f"**{current_session['title']}**")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("질문을 입력해주세요:")

    if prompt:
        user_message = {"role": "user", 
                        "content": prompt, 
                        "timestamp": datetime.now()}
        st.session_state.chat_history.append(user_message)
        current_session["messages"].append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)


        try:
            agent_input = {
                    "input": prompt,
                    "chat_history": st.session_state.chat_history,
                    "intermediate_steps": [],
                    "agent_scratchpad": "",
                    "genres": ", ".join(config.unique_genres),
                    "actors": ", ".join(cast_list),
            }
            agent_outcome = run_agent(agent_input)

            extracted_actor = agent_outcome.get("actor")
            extracted_genre = agent_outcome.get("genre")

            filter_condition = "현재" in prompt or "실시간" in prompt or "상영중" in prompt
            booking_condition = "예매" in prompt
            if extracted_actor:
                st.session_state.last_actor = extracted_actor
            if extracted_genre:
                st.session_state.last_genre = extracted_genre
            actor = st.session_state.last_actor
            genre = st.session_state.last_genre

            if filter_condition:
                top_titles = recommender.score(actor, genre)

                if not top_titles:
                    st.markdown("현재 상영 중인 조건에 맞는 뮤지컬이 없습니다.")

                else:
                    current_date = datetime.now().strftime('%Y.%m.%d')
                    matched_recommendations = add_genre_data[
                        (add_genre_data['title'].isin(top_titles)) &
                        (add_genre_data['end_date'] > current_date)
                    ]

                    if matched_recommendations.empty:
                        with st.chat_message("assistant"):
                            message = "추천 조건에 맞는 뮤지컬 정보를 찾을 수 없습니다."
                    else:
                        active_titles = matched_recommendations['title'].tolist()
                        st.session_state.active_titles = active_titles
                        st.markdown("현재 상영 중인 추천 뮤지컬 목록:\n")
                        recommendation_message = ""
                        recommendation_img = ""
                        for _, row in matched_recommendations.iterrows():
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(row['poster'], width=100, caption=row['title'])
                            with col2:    
                                markdown_message = (
                                    f"- **{row['title']}**\n"
                                    f"  - 장소: {row['place']}\n"
                                    f"  - 배우: {row['cast']}\n"
                                    f"  - 장르: {row['genre']}\n"
                                    f"  - 종료일: {row['end_date']}\n"
                                    f"  - 시간: {row['time']}\n"
                                    f"  - 가격: {row['ticket_price']}\n\n"
                                )
                                st.markdown(
                                    f"- **{row['title']}**\n"
                                    f"  - 장소: {row['place']}\n"
                                    f"  - 배우: {row['cast']}\n"
                                    f"  - 장르: {row['genre']}\n"
                                    f"  - 종료일: {row['end_date']}\n"
                                    f"  - 시간: {row['time']}\n"
                                    f"  - 가격: {row['ticket_price']}\n\n"
                                )
                                recommendation_message += markdown_message

                        assistant_message = {
                            "role": "assistant",
                            "content": recommendation_message,
                            "timestamp": datetime.now(),
                        }
                        st.session_state.chat_history.append(assistant_message)
                        current_session["messages"].append(assistant_message)
                        st.markdown("예매 링크를 안내해드릴까요?")
            
            elif booking_condition:
                active_titles = st.session_state.active_titles
                booking_message = ""
                if not active_titles:
                    st.markdown("추천된 뮤지컬이 없습니다. 먼저 추천을 받아주세요.")
                else:
                    st.markdown("**예매 가능한 링크를 가져옵니다...**")
                    for title in active_titles:
                        url = fetch_interpark_ticket_url(title)
                        if url:
                            markdown_booking = f"- [{title} 예매하기]({url})\n"
                            booking_message += markdown_booking
                        else:
                            st.markdown(f"- **{title}**: 예매 링크를 찾을 수 없습니다.")

                    with st.chat_message("assistant"):
                        st.markdown(booking_message)

                    assistant_message = {
                        "role": "assistant",
                        "content": markdown_booking,
                        "timestamp": datetime.now(),
                    }
                    st.session_state.chat_history.append(assistant_message)
                    current_session["messages"].append(assistant_message)

            else:

                recommendations = recommender.recommend(extracted_actor,extracted_genre)
                recommendations = recommendations.iloc[::-1]

                st.session_state["recommendations"] = recommendations
                recommendation_message = ""
                recommendation_img = ""
                for idx, (_, row) in enumerate(recommendations.iterrows()):
                        if idx < 3:
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(row['poster'], width=100, caption=row['title'])
                            with col2:
                                recommendation_message += (
                                    f"**{row['title']}**\n"
                                    f"- 장소: {row['place']}\n"
                                    f"- 배우: {row['cast']}\n"
                                    f"- 가격: {row['ticket_price']}\n\n"
                                ) 
                                st.markdown(
                                    f"**{row['title']}**\n"
                                    f"- 장소: {row['place']}\n"
                                    f"- 배우: {row['cast']}\n"
                                    f"- 가격: {row['ticket_price']}\n\n"
                                )
                        else:
                            pass

                assistant_message = {
                    "role": "assistant",
                    "content": recommendation_message,
                    "timestamp": datetime.now(),
                }
                st.session_state.chat_history.append(assistant_message)
                current_session["messages"].append(assistant_message)

                follow_up_message = "현재 상영중인 뮤지컬을 추천해드릴까요?"
                with st.chat_message("assistant"):
                    st.markdown(follow_up_message)

                follow_up_message_data = {
                    "role": "assistant",
                    "content": follow_up_message,
                    "timestamp": datetime.now(),
                }
                st.session_state.chat_history.append(follow_up_message_data)
                current_session["messages"].append(follow_up_message_data)
        
        except Exception as e:
            if extracted_actor == "None":
                    error_message = f"배우를 함께 입력해주세요."
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
            if extracted_genre == "None":
                    error_message = f"장르를 함께 입력해주세요."
                    with st.chat_message("assistant"):
                        st.markdown(error_message)

            else:
                error_message = f"개발자를 갈아넣어 더 성장하겠습니다."
                with st.chat_message("assistant"):
                    st.markdown(error_message)

                error_message_data = {
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": datetime.now(),
                }
                st.session_state.chat_history.append(error_message_data)
                current_session["messages"].append(error_message_data)
else:
    st.write("안녕하세요 ☺️ 뮤지컬 관람 계획 중이신가요? \n\n왼쪽에서 채팅방을 선택하거나 새로 시작하세요.")