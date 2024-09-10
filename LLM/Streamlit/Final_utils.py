import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import streamlit as st
import numpy as np
import random
from dotenv import load_dotenv
####################
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
########################################
import torch
import torchvision.models as models
import torch.nn as nn
import streamlit as st
from torchvision.models import DenseNet201_Weights, VGG19_Weights
########################################
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.schema import SystemMessage, HumanMessage
from langchain_community.llms import OpenAI
from PIL import ExifTags
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from ultralytics import YOLO
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import(
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableMap
)
from dotenv import load_dotenv
load_dotenv()

################################################### Vector DB #############################################
# Studies DB와 Products DB
studies_store = Chroma(
collection_name="studies",
embedding_function=OpenAIEmbeddings(),
persist_directory= r'D:\MulCam\팀프로젝트\최종프로젝트0712_0910\Code\Code\LLM\VectorDB\chroma_studies_v3_0905'
)

prod_store = Chroma(
collection_name='product_list_v3_0909',
embedding_function=OpenAIEmbeddings(), 
persist_directory= r'D:\MulCam\팀프로젝트\최종프로젝트0712_0910\Code\Code\LLM\VectorDB\chroma_product_list_v3_0909'
)

studies_retriever = studies_store.as_retriever(search_kwargs={'k' : 3})
prod_retriever = prod_store.as_retriever(search_kwargs={'k' : 10})

############################################### 이미지 모델 정의 함수 #############################################
@st.cache_resource 
def load_models():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/team2_yolov5m_100.pt', force_reload=True)
    yolo_model.imgsz = 416
    yolo_model.conf = 0.3
    yolo_model.iou = 0.5
    yolo_model.agnostic = True
    yolo_model.stride = 1
    yolo_model.max_det = 1000
    yolo_model.line_thickness = 2

    model_class_id_map = {
        'pigmentation_forehead': 1,  # 이마
        'pigmentation_cheek_l': 5,   # 왼쪽 볼
        'pigmentation_cheek_r': 6,   # 오른쪽 볼
        'wrinkle_perocular_r' : 4,   # 오른쪽 눈가
        'wrinkle_perocular_l' : 3,    # 왼쪽 눈가
        'wrinkle_forehead': 1,       # 이마
        'wrinkle_glabellus': 2,       # 턱
        'chin_sagging' : 8,          # 턱
        'l_cheek_pore' : 5,          # 왼쪽 볼
        'r_cheek_pore' : 6           # 오른쪽 볼
    }

    # Load DenseNet201 and VGG19 weights
    densenet_weights = DenseNet201_Weights.DEFAULT
    vgg19_weights = VGG19_Weights.DEFAULT

    class DenseNet201_VGG19_Ensemble(nn.Module):
        def __init__(self, num_classes, drop_out):
            super(DenseNet201_VGG19_Ensemble, self).__init__()
            self.densenet = models.densenet201(weights=densenet_weights)
            densenet_features = self.densenet.classifier.in_features
            self.densenet.classifier = nn.Identity()
            self.vgg = models.vgg19(weights=vgg19_weights)
            vgg_features = self.vgg.classifier[0].in_features
            self.vgg.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(densenet_features + vgg_features, 1024),
                nn.ReLU(),
                nn.Dropout(p=drop_out),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=drop_out),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            densenet_features = self.densenet(x)
            vgg_features = self.vgg(x)
            combined_features = torch.cat((densenet_features, vgg_features), dim=1)
            output = self.classifier(combined_features)
            return output
        
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        

    model_settings = {
        'pigmentation_forehead': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/forehead_pigmentation.pth',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'pigmentation_cheek_l': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/cheek_pigmentation.pth',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'pigmentation_cheek_r': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/cheek_pigmentation.pth',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'wrinkle_perocular_r': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/perocular_wrinkle.pth',
            'num_classes': 2,
            'dropout_rate': 0.3
        },
        'wrinkle_perocular_l': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/perocular_wrinkle.pth',
            'num_classes': 2,
            'dropout_rate': 0.3
        },
        'wrinkle_forehead': {
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/forehead_wrinkle.pth',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'wrinkle_glabellus':{
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/wrinkle_glabellus.pth',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'chin_sagging':{
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/chin_sagging.pt',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'l_cheek_pore':{
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/cheek_pore.pt',
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'r_cheek_pore':{
            'path': 'D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/cheek_pore.pt',
            'num_classes': 2,
            'dropout_rate': 0.5
        }
    }
    return yolo_model, model_class_id_map, model_settings, DenseNet201_VGG19_Ensemble, transform


############################################### 등급 분류 함수(Ensemble) #############################################

def cropped_img(annotation, resnet_model, results, model_class_id, img):
    detection_data = results.xyxy[0].cpu().numpy()
    json_results = []
    max_confidence = -1
    best_box = None

    # 크롭된 이미지를 저장할 디렉토리 생성 (존재하지 않으면 생성)
    # Path(img_save_dir).mkdir(parents=True, exist_ok=True)

    for *box, conf, cls in detection_data:
        class_id = int(cls)
        if class_id == model_class_id:
            if conf > max_confidence:
                max_confidence = conf
                best_box = box

    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        # 이미지를 크롭합니다 (PIL로 변환 후 처리)

        cropped_img_pil = img.crop((x1, y1, x2, y2))

        # 이미지 저장 경로 설정
        # img_save_path = f"{img_save_dir}/{annotation}.png"
        # cropped_img_pil.save(img_save_path)  # 크롭된 이미지 저장
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 이미지 분석 (모델 적용)
        input_tensor = transform(cropped_img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = resnet_model(input_tensor)
            predicted_class = torch.argmax(prediction, dim=1).item()

        json_results.append({
            "class": annotation,
            "predicted_class": predicted_class
        })

    return json_results

############################################### 모델 파라미터 로드 함수 #############################################

def rename_state_dict_keys(state_dict, prefix="vgg"):
    renamed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vgg19."):
            new_key = key.replace("vgg19.", f"{prefix}.")
            renamed_state_dict[new_key] = value
        else:
            renamed_state_dict[key] = value
    return renamed_state_dict

############################################### 이미지 분석 결과 생성 #############################################

def create_prompt(json_data):
    class_name_map = {
        'pigmentation_forehead': '이마 색소 침착',
        "pigmentation_cheek_l": "왼쪽 볼 색소 침착",
        "pigmentation_cheek_r": "오른쪽 볼 색소 침착",
        "l_cheek_pore": "왼쪽 볼 모공",
        "r_cheek_pore": "오른쪽 볼 모공",
        'chin_sagging': "턱 탄력",
        'wrinkle_forehead':'이마 주름',
        'wrinkle_perocular_r':'오른쪽 눈가 주름',
        'wrinkle_perocular_l':'왼쪽 눈가 주름',
        'wrinkle_glabellus': '미간 주름'
    }

    results = json_data
    analysis = {
        "pigmentation": [],
        "wrinkle": [],
        "chin": [],
        "pore": []
    }
    
    for item in results:
        if "pigmentation" in item["class"]:
            analysis["pigmentation"].append(item)
        elif "wrinkle" in item["class"]:
            analysis["wrinkle"].append(item)
        elif "chin" in item["class"]:
            analysis["chin"].append(item)
        elif "pore" in item["class"]:
            analysis["pore"].append(item)

    detailed_result = []
    for key, value in analysis.items():
        for item in value:
            if item["predicted_class"] == 1:
                part = class_name_map.get(item["class"], item["class"])
                detailed_result.append(part)
        
    return detailed_result

############################################### llm 함수 ###################################################

def get_llm():
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.56,
    )
    return llm

################################################ 피부 분석 결과 도출 함수 ###################################################

def display_results_and_summary(detailed_result, session_id):
# 결과를 GRID 형식으로 출력하고 요약문 생성
    detailed_text = "\n".join(detailed_result)
    if detailed_result:
        prompt = (
            f"""
            분석 결과를 작성하기 전, 아래의 형식에 따라 답변을 생성해 주세요 [FORMAT1]
            
            FORMAT1:
            {session_id}님은 다음 부위의 피부 관리가 필요합니다.
            FORMAT1의 답변을 생성한 후에 아래의 내용을 진행해 주세요
            사용자가 얼굴 사진에 대한 분석을 받았습니다. 아래는 분석해준 결과입니다. \n\n
            {detailed_text}
            분석한 결과를 이용하여 아래의 형식에 따라 답변을 생성해 주세요 [FORMAT2]
            FORMAT2:
            - > **왼쪽 볼 모공**
            - > **이마 주름**
            - > **오른쪽 볼 색소침착**
            1. FORMAT 형식 답변 외에는 새로운 문장을 작성하지 마세요.
            2. 메시지에서 불필요한 문장 부호를 삭제해 주세요.
            3. 챗봇 형식에 맞게 사용자에게 쉽게 전달할 수 있도록 답변해 주세요.
            그 후 위 분석 결과를 한 문장으로 요약해서 사용자에게 전달해 주세요.
            출력 결과는 '요약된 결과:'라는 문구나 대괄호([])를 포함하지 않고, 순수한 문장만 생성해 주세요.
            예를 들어, 분석 결과가 '이마 색소침착, 왼쪽 볼 색소침착, 턱 탄력 부족'이라면
            '이마와 왼쪽 볼에 색소침착이 나타나고 턱에 탄력이 부족한 상태입니다.'라는 식으로 답변해 주세요.
            그 다음 '해당 부위에 맞는 화장품 추천과 피부 관리 방법을 제공해드릴 수 있어요. 그 외 궁금하신 사항이 있으시면 입력해주세요.' 라고 답변을 남겨주세요.
            답변할때 온점 . 을 기준으로 줄바꿈해서 표시해주세요.
            요약 결과는 분석 결과 아래에 출력해 주세요.
            """
        )
    else:
        prompt = (
                f"""
                분석 결과를 작성하기 전, 아래의 형식에 따라 답변을 생성해 주세요 [FORMAT1]
                FORMAT1:
                {session_id}님은 피부관리를 잘 하고 계시네요.
                앙투라지는 화장품 추천과 피부 관리 방법을 제공해드릴 수 있습니다.
                추가적인 질문이 있으시면 입력해주세요.'
                1. FORMAT 형식 답변 외에는 새로운 문장을 작성하지 마세요.
                2. 메시지에서 불필요한 문장 부호를 삭제해 주세요.
                3. 챗봇 형식에 맞게 사용자에게 쉽게 전달할 수 있도록 답변해 주세요.
                """
            )
    llm = get_llm()
    response = llm([HumanMessage(content=prompt)])
    # 'assistant'가 답변하는 것 저장
    st.session_state['messages'].append(ChatMessage(role='assistant', content=response.content))

    # 해당 사용자 session의 history에 저장
    history = get_session_history(session_id)
    history.add_ai_message(response.content)
    

################################################ stream, 대화기록 출력 함수 ###################################################

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def print_messages():    
    # 새로고침하기 전에 'messages'에 있는 내용 보여주기
    if 'messages' in st.session_state and len(st.session_state['messages']) > 0:
        # 'user'가 입력한 내용은 'user' 아이콘과 함께 나가야 하고, 'assistant'가 작성한 내용은 'assistant' 아이콘과 함께 나가야 한다.
        for chat_message in st.session_state['messages']:
            st.chat_message(chat_message.role).write(chat_message.content)

################################################ 세션 ID를 기반으로 세션 기록을 가져오는 함수 ###################################################

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state['store'][session_id] = ChatMessageHistory()
    return st.session_state['store'][session_id]  # 해당 세션 ID에 대한 세션 기록 반환

######################################## 질문 분리(추천, 관리, 기타) / 키워드 반환 함수 ###################################################

def classify_and_extract_keywords(question):
    model = ChatOpenAI(temperature=0.56, model_name="gpt-4o-mini")
    parser = JsonOutputParser()
    prompt = PromptTemplate.from_template(
        """주어진 질문을 `피부관리방법`, `피부관련제품추천`, `피부관련제품`, 또는 `기타` 중 하나로 분류하세요. 
        해당 질문의 키워드도 함께 추출하세요.

        <question>{question}</question>
        JSON 형식으로 반환하세요:
        {{
            "Classification": "피부관리방법",
            "Keyword": ["피부", "관리", "방법"]
        }}
        추가로, 질문이 "더마펌흔적끝토닝세럼 어때요"과 같이 특정 제품명이 포함되면, 
        "제품명"이라는 글자와 함께 추출해 주세요. 
        
        {{
            "Classification": "피부관련제품",
            "Keyword": ["제품", "더마펌흔적끝토닝세럼", "제품명"]
            
            }}
            
        "다른 제품을 추천 해줘"와 같이 다른이 들어간 문장이면, 
        다시 제품을 추천 해줘.
        {{
            "Classification": "다른제품추천",
            "Keyword": ["다른", "제품" "추천", "방법"]
        }}
        
        질문이 문장 그대로 "제품 추천해줘", "제품 추천", "제품 추천 해줘" 라고 들어오면, 
        "상세한 내용을 추가로 작성해주세요"라는 답변을 제공합니다.
        {{
            "Classification": "추가정보필요",
            "Keyword": ["제품", "추천", "상세정보"]
        }}

        """
    )

    route_chain = {"question": RunnablePassthrough()} | prompt | model | parser
    return route_chain.invoke(question)

######################################## 연구 논문 검색 함수 ###################################################

def studies_chain(question, history, model):
    relevant_docs = studies_retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    studies_prompt = PromptTemplate.from_template(
        """당신은 피부과 전문의입니다.
        채팅 내용을 통해 증상에 따라 피부 관리 방법을 묻는다면, 채팅 기록으로 질문에 답하세요.
        다음 정보를 바탕으로 사용자의 질문에 답하세요:

        {context}
        이전 채팅 기록: {history}

        질문: {question}
        답변:"""
    )
    chain = studies_prompt | model | StrOutputParser()
    return chain.invoke({"question": question, "context": context, "history": history})

######################################## 제품 추천 시 랜덤 추출 함수 ###################################################

def prod_chain(question, history, model):
    documents = prod_retriever.invoke(question)
    
    # '제품명'이 질문에 포함된 경우 단일 제품만 선택
    if "제품명" in question:
        selected_docs = documents[:1]  # 첫 번째 제품만 선택
        # 단일 제품에 대한 요약문만 생성
        response = generate_single_product_response(selected_docs[0], question, history, model)
    else:
        selected_docs = random.sample(documents, min(3, len(documents)))
        responses = []
        for doc in selected_docs:
            response = generate_product_response(doc, question, history, model)
            responses.append(response)
        response = "\n\n".join(responses)
    
    return response

######################################## 제품 추천 함수  ###################################################

def generate_product_response(doc, question, history, model):
    # 딕셔너리로 각 제품의 정보를 저장
    context = {
        "picture": doc.metadata['picture'],
        "name": doc.metadata['name'],
        "brand": doc.metadata['brand'],
        "price": doc.metadata['price'],
        "ingredient": doc.metadata['ingredient'],
        "link": doc.metadata['link'],
        "Summary": doc.metadata['Summary']
    }
    
    picture = doc.metadata['picture']
    link = doc.metadata['link']
    
    prompt = PromptTemplate.from_template(
    """
    당신은 화장품 매장에서 10년의 경험을 가진 매니저입니다. 추천을 하기 전에, 처음에만 "다음 화장품 제품을 추천드리고 싶습니다"라는 인사말로 시작해야 합니다.
    
    
    아래 형식에 따라 답변을 생성하세요 [FORMAT].
    피부 화장품 정보가 사전형 데이터로 제공됩니다:
    {context}

    FORMAT:
    ![제품 이미지]({picture}) 
    > ####  name  
    > **브랜드** : brand  
    > **가격** : price원  
    > **링크** : [구매 링크]({link})  

    > **리뷰요약:**  
    > Summary

    1. Summarize the product in three concise sentences.
    2. Add '원' after the price (which should be an integer).
    3. Remove any unnecessary punctuation from the message.
    4. Present the brand, price, and link on separate lines.
    """
    )
    
    chain = prompt | model | StrOutputParser()
    return chain.invoke({'context': context, 'picture': picture, 'link': link, 'question': question, 'history': history})

def extract_keywords(question):
    # 간단한 키워드 추출 방법 (예: 사용자가 원하는 키워드 목록을 추출)
    keywords = []
    if '가격' in question:
        keywords.append('가격')
    if '성분' in question:
        keywords.append('성분')
    if any(keyword in question for keyword in ['전성분', '모든 성분']):
        keywords.append('성분')
    if any(keyword in question for keyword in ['효과', '좋아', '어디에']):
        keywords.append('효과')
    return ', '.join(keywords)

######################################## 단일 제품 정보 함수  ###################################################

def generate_single_product_response(doc, question, history, model):
    # 딕셔너리로 각 제품의 정보를 저장
    context = {
        "picture": doc.metadata['picture'],
        "name": doc.metadata['name'],
        "brand": doc.metadata['brand'],
        "price": doc.metadata['price'],
        "ingredient": doc.metadata['ingredient'],
        "link": doc.metadata['link'],
        "Summary": doc.metadata['Summary']
    }   
    
    keywords = extract_keywords(question)    
    
    prompt = PromptTemplate.from_template(
    """
    당신은 화장품 매장에서 10년의 경험을 가진 매니저입니다. 추천을 하기 전에, 처음에만 "다음 화장품 제품에 대해 알려 드리겠습니다"라는 인사말로 시작해야 합니다.
    
    아래 형식에 따라 답변을 생성하세요 [FORMAT].
    피부 화장품 정보가 사전형 데이터로 제공됩니다:
    {context}
    
    Previous chat history: {history}
    
    FORMAT:

    > **리뷰요약:**  
    > Summary
    > 사용자의 질문과 관련된 내용으로 요약해 주세요. 예를 들어, 질문에 '가격'이 포함되어 있으면 가격에 대한 정보를 강조하고, '성분'이 포함되어 있으면 주요 성분과 그 효과를 설명하십시오.  
    '전성분'이나 '모든 성분'이 포함되어 있으면 해당 제품의 모든 성분을 제공해주세요.     
     - 만약 '효과'가 키워드에 포함되어 있으면 아래의 Summary를 그대로 제공합니다:
    > {Summary}
    
    - 질문에 나온 키워드({keywords})를 바탕으로 제품을 간결한 세 문장으로 요약해보세요.
    - 적절한 줄 바꿈을 사용하여 명확하고 읽기 쉬운 형식으로 요약을 제공합니다.
    - 불필요한 구두점을 피하고 텍스트가 부드럽고 일관성이 있는지 확인하십시오.
    - 명확성과 가독성을 위해 필요한 경우 글머리 기호를 사용하세요.
    """
    )
    
    chain = prompt | model | StrOutputParser()
    return chain.invoke({'context': context,'keywords': keywords, 'question': question, 'history': history})


################################################# 기타 질문 ####################################################################

# General Chain (기타 질문)
def general_chain(question, history, model):
    general_prompt = PromptTemplate.from_template(
        """
        당신은 피부 전문가입니다. 주어진 질문이 피부 관련 주제인지 확인한 후에 적절히 답변하세요. 
        Previous Chat History를 잘 참고해서 응답해 주세요.
        만약 질문이 피부 관련 제품, 피부 관리 방법 또는 일반적인 인사/자기소개와 같은 질문이 아니라면 다음과 같이 응답하세요.:
        
        Previous Chat History: {history}
        
        피부와 화장품과 관련 없는 질문일 경우, 다음 기준에 따라 답변하세요:
        1. 질문이 스킨케어와 전혀 관련이 없을 경우:
        "저는 스킨케어 챗봇이므로 이 질문에 답변할 수 없습니다. 피부 고민이나 제품 추천에 대해 언제든지 질문해 주세요."라고 답변하세요.
        
        2. 부적절하거나 유해한 내용일 경우:
        예의 바르게 대화를 거절하고, 스킨케어 관련 주제로 대화를 전환하세요.
        
        3. 모든 경우에 대해:
        항상 친절하고 전문적인 어조를 유지하세요.
        개인 정보를 요청하거나 공유하지 마세요.
        의료 진단이나 치료 방법을 제공하지 마세요.
        
        4. classification가 "추가정보필요"일 경우:
        "좀 더 상세한 내용을 작성해주세요. 더 성심껏 답변해드리겠습니다."라고 답변하세요.
         
        피부 관련 질문일 경우, 다음 기준에 따라 답변하세요:
        
        1. **피부 관련 제품 추천**: 질문이 특정 피부 고민에 대한 제품 추천을 요청하는 경우, 피부 타입에 맞는 적절한 제품을 추천합니다.
        1. **피부 관련 제품 추천**: 질문이 특정 제품에 대한 정보를 요청하는 경우, 가격, 요약 정보등을 제공합니다.
        1. **피부 관련 제품 추천**: 질문이 다른 제품 추천을 요청하는 경우, 다시 새로운 제품을 추천합니다.
        2. **피부 관리 방법**: 질문이 피부 관리 방법에 대한 경우, 피부 상태를 개선할 수 있는 관리 방법을 구체적으로 설명합니다.
        3. **일반 질문**: 인사나 자기소개, 대화 상에서의 간단한 응답 요청일 경우, 간단한 답변을 제공합니다.
        
        질문: {question}
        답변:
        """
    )
    
    chain = general_prompt | model | StrOutputParser()
    return chain.invoke({"question": question, "history": history})

# 전체 과정 실행 함수
def process_user_query(question, user):

    model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",
                        streaming=True, callbacks=[StreamHandler(st.empty())])
    # 1. 질문 분류 및 키워드 추출
    classified_response = classify_and_extract_keywords(question)
    classification = classified_response['Classification']
    
    # Get the session chat history (세션기록 가져오기)
    history = get_session_history(user).messages

    # 2. 질문에 따라 적절한 체인 호출
    if classification == "피부관리방법":
        answer = studies_chain(question, history, model)
    elif classification == "피부관련제품추천":
        answer = prod_chain(question, history, model)
    elif classification == "다른제품추천":
        answer = prod_chain(question, history, model)  # 다시 제품 추천 실행
    elif classification == "추가정보필요":
        answer = general_chain(question, history, model)
    else:
        answer = general_chain(question, history, model)

    return answer



