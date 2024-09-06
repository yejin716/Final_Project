import torch
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import torch.nn as nn
import streamlit as st
import json
import numpy as np
import os
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
from langchain.schema import SystemMessage, HumanMessage
from PIL import ExifTags
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory  
 
store = {}

 
#이미지 모델 정의 함수  
@st.cache_resource 
def load_models():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/MulCam/팀프로젝트/최종프로젝트0712_0910/Code/Code/LLM/models/team2_yolov5m_100.pt', force_reload=True)
    yolo_model.imgsz = 416
    yolo_model.conf = 0.3
    yolo_model.iou = 0.5
    yolo_model.agnostic = True
    # yolo_model.stride = 1
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


#크롭된 이미지를 ensemble모델에 넣어 등급 분류 함수 
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


# 모델 파라미터 로드 함수 
def rename_state_dict_keys(state_dict, prefix="vgg"):
    renamed_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vgg19."):
            new_key = key.replace("vgg19.", f"{prefix}.")
            renamed_state_dict[new_key] = value
        else:
            renamed_state_dict[key] = value
    return renamed_state_dict


#관리가 필요한 부위 선별 함수 
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


#llm 함수 
def get_llm():
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0.5,
    )
    return llm


def display_results_and_summary(detailed_result, user_name):
# 결과를 GRID 형식으로 출력하고 요약문 생성
    detailed_text = "\n".join(detailed_result)
    if detailed_result:
        prompt = (
            f"""
            분석 결과를 작성하기 전, 아래의 형식에 따라 답변을 생성해 주세요 [FORMAT1]
            
            FORMAT1:
            {user_name}님은 다음 부위의 피부 관리가 필요합니다.
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
                {user_name}님은 피부관리를 잘 하고 계시네요.
                앙투라지는 화장품 추천과 피부 관리 방법을 제공해드릴 수 있습니다.
                추가적인 질문이 있으시면 입력해주세요.'
                1. FORMAT 형식 답변 외에는 새로운 문장을 작성하지 마세요.
                2. 메시지에서 불필요한 문장 부호를 삭제해 주세요.
                3. 챗봇 형식에 맞게 사용자에게 쉽게 전달할 수 있도록 답변해 주세요.
                """
            )
    llm = get_llm()
    response = llm([HumanMessage(content=prompt)])
    # AI 응답 결과를 출력
    st.chat_message("assistant").write(response.content)
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})
    
    
