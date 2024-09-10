import streamlit as st
import os
import sys
from dotenv import load_dotenv
from streamlit_navigation_bar import st_navbar
load_dotenv()
import base64
# pages.py 경로 추가
sys.path.insert(0, 'D:\MulCam\팀프로젝트\최종프로젝트0712_0910\Code\Code\LLM\Streamlit\pages.py')  # `pages.py` 파일이 위치한 경로로 설정

# 페이지 내용 함수 불러오기
import pages as pg




# 네비게이션 바 스타일 설정
pages =["Home", "ChatBot", "Our Story"]

styles = {
    "nav": {
        "background-color": "rgb(86, 130, 86)",
        "height": "80px",  # 네비게이션 바의 높이
        "font-family": 'Helvetica',
    },
    "div": {
        "max-width": "40rem",  # 네비게이션 바의 너비
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(242, 246, 235)",
        "margin": "0 0.125rem",
        "padding": "0.75rem 1rem",  # 네비게이션 항목의 패딩을 조정
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}


# 네비게이션 바 렌더링
page = st_navbar(pages, styles=styles)

# st.markdown('<h1 class="title-left">Entourage</h1>', unsafe_allow_html=True)

# 페이지에 맞는 함수 호출
functions = {
    "Home": pg.show_home,
    "ChatBot": pg.show_chatbot,
    "Our Story": pg.show_our_story

}

# 선택된 페이지에 맞는 함수 실행
if page in functions:
    functions[page]()

# 제목 스타일 추가
st.markdown("""
    <style>
    .title-left {
        font-size: 3.5rem;  /* 글자 크기 조정 */
        margin-top: -30px;  /* 제목의 위쪽 여백 조정 */
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)