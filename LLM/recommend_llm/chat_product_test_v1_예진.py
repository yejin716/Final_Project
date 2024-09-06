import os
import streamlit as st
from dotenv import load_dotenv
import random
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent
from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from chromadb import Client as ChromaClient
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import re
import os
import pandas as pd 
load_dotenv()
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

################################라이브러리##########################################

# LLM 및 Chroma DB 클라이언트 초기화
llm = OpenAI(model_name="gpt-4o-mini")
database = Chroma(
    collection_name='product_list', 
    embedding_function=OpenAIEmbeddings(), 
    persist_directory='../VectorDB/chroma_product_list_v1_0831'
)

################################DB Load#############################################

# retriever 객체 초기화
retriever = database.as_retriever()


################################Retriever 초기화#############################################


st.set_page_config(page_title='TEAM.2 앙투라지', page_icon="🏆",initial_sidebar_state="collapsed")
st.title('🤖DERMA A.I BOT')
st.caption('👩‍🔬얼굴 사진을 올려주세요. 피부분석에 관련된 고민을 들어드립니다.')

# Session State also supports attribute based syntax
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

##################################화장품 이미지 url##########################################
df = pd.read_csv('./cosmetics.csv')

def get_image_url(name, df):
    match = df[df['name'] == name]
    if not match.empty:
        return match['picture'].values[0]  # picture 컬럼의 값을 반환
    else:
        return ''  # 일치하는 행이 없으면 빈 문자열 반환

##################################RAG 코드##########################################
def get_ai_message(query):

    # LLM을 이용한 키워드 추출 함수
    def extract_keywords_from_text(query):
        # LLM에게 키워드를 추출하도록 프롬프트를 생성
        prompt_template = "다음 문장에서 핵심 키워드를 추출해줘: '{query}'"
        prompt = PromptTemplate(input_variables=["query"], template=prompt_template)
        
        llm = OpenAI()
        # LLM에 프롬프트를 전달하여 키워드 추출
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke(query)
        # 키워드를 쉼표로 구분한 결과를 리스트로 변환
        keywords = [kw.strip() for kw in response.split(",")]
        return keywords

    # Chroma DB에서 검색하는 함수
    def search_chroma_db(keywords, retriever):
        # 키워드들을 조합하여 검색 수행
        query = " ".join(keywords)  # 키워드들을 공백으로 구분하여 하나의 쿼리로 결합
        results = retriever.invoke(query)
        return results

    # 응답을 생성하는 함수
    def generate_response(documents):
        responses = []

        # 랜덤으로 
        selected_docs = random.sample(documents, min(3, len(documents)))

        for idx, doc in enumerate(selected_docs, start=1):
            image_url = get_image_url(doc.metadata['name'], df)
            
            # 딕셔너리로 각 제품의 정보를 저장
            context = {
                "name": doc.metadata['name'],
                "brand": doc.metadata['brand'],
                "price": doc.metadata['price'],
                "url": doc.metadata['url'],
                "image_url": image_url,
                "Summary": doc.metadata['Summary']
            }

            url = doc.metadata['url']

            # 프롬프트 템플릿 설정
            if idx == 1:
                template = """
                            당신은 화장품 가게에서 10년간 근무한 점장입니다. 추천을 하기전 맨 처음에만 "다음과 같은 화장품을 추천해드리겠습니다"의 인사로 시작하여야 합니다.
                            
                            아래의 형식에 따라 답변을 생성해 주세요 [FORMAT].
                            You are provided with dictionary type data which is the information of a dermal cosmetic product:
                            {context}

                            FORMAT:
                            ![제품 이미지]({image_url}) 
                            > ####  name  
                            > **브랜드** \n brand  
                            > **가격** \n price원  
                            > **링크** \n [구매 링크]({url})  

                            > **리뷰요약:**  
                            > Summary

                            1. 'Summary'를 3문장으로 요약한 후 답변을 생성해 주세요.
                            2. 가격(int) 뒤에 '원'을 붙여 주세요.
                            3. 메시지에서 불필요한 문장 부호를 삭제해 주세요.
                            4. 브랜드, 가격, 링크에 해당 하는 값을 줄바꿈해서 답변을 생성해 주세요.
                            """
            else:
                
                template = """
                            아래의 형식에 따라 답변을 생성해 주세요 [FORMAT].
                            You are provided with dictionary type data which is the information of a dermal cosmetic product:
                            {context}

                            FORMAT:
                            ![제품 이미지]({image_url}) 
                            > ####  name  
                            > **브랜드** \n brand  
                            > **가격** \n price원  
                            > **링크** \n [구매 링크]({url})  

                            > **리뷰요약:**  
                            > Summary


                            1. 'Summary'를 3문장으로 요약한 후 답변을 생성해 주세요.
                            2. 가격(int) 뒤에 '원'을 붙여 주세요.
                            3. 메시지에서 불필요한 문장 부호를 삭제해 주세요.
                            4. 브랜드, 가격, 링크에 해당 하는 값을 줄바꿈해서 답변을 생성해 주세요.
                            """


            prompt = PromptTemplate.from_template(template)

            model =  ChatOpenAI(model_name = 'gpt-4o')

            output_parser = StrOutputParser()


            chain = prompt | model | output_parser
            response = chain.invoke({'context': context, 'url':url, 'image_url': image_url})

            responses.append(response)
        
        return responses

    # 쿼리 처리 및 제품 추천
    keywords = extract_keywords_from_text(query)

    documents = search_chroma_db(keywords, retriever)

    ai_message = generate_response(documents)

    return ai_message



if user_question := st.chat_input(placeholder='피부고민이 있어요? 궁금해요.'):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user', 'content':user_question})

    ai_message = get_ai_message(user_question)
    with st.chat_message('ai'):
        for response in ai_message:
            st.markdown(response)
    st.session_state.message_list.append({'role':'ai', 'content':ai_message})