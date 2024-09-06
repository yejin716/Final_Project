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
################################################
from retriever_agent import run_retriever_agent
from Entourage_chatbot_def import load_models, cropped_img, rename_state_dict_keys,create_prompt
from Entourage_chatbot_def import get_llm, display_results_and_summary

# 환경 변수 설정
load_dotenv()


  
#######################################################################################################################
st.set_page_config(page_title='TEAM.2 앙투라지', page_icon="🏆") #,initial_sidebar_state="collapsed"
st.title('🤖DERMA A.I BOT')
st.subheader('피부 분석을 통해 피부 고민을 들어드립니다~~')
st.caption("사진을 업로드하시거나 평소 피부에 대해 궁금하셨던 부분이 있으시다면 입력해주세요.")
#######################################################################################################################
#yolov5, ensemble 모델 정의 
yolo_model, model_class_id_map, model_settings, DenseNet201_VGG19_Ensemble, transform = load_models()
#######################################################################################################################
#이미지 업로드 #
###############

# 세션 상태 초기화 함수
def reset_session_state():
    st.session_state.name_entered = False
    st.session_state.json_saved = False
    st.session_state.json_data = None
    
with st.sidebar:
    st.markdown('✅**메이크업을 하지 않은사진**을 올려주세요.', unsafe_allow_html=True)
    st.markdown('✅**깔끔한 배경**에서 **정면**으로 찍으시면 더 정확하게 분석해드립니다.', unsafe_allow_html=True)
    file = st.file_uploader('이미지를 업로드하세요', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

    # 만약 사용자가 새로 이미지를 업로드했다면 세션 상태를 초기화
    if file is not None:
        # reset_session_state()  # 세션 상태 초기화
        image = Image.open(file)
        
        # EXIF 데이터에서 회전 정보를 가져와서 이미지 회전
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()

            if exif is not None:
                orientation = exif.get(orientation, 1)

                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # EXIF 데이터가 없거나 오류가 발생한 경우, 회전하지 않고 원본 이미지를 사용
            pass
            
        st.session_state.image = image
        st.session_state.image_uploaded = True
        st.session_state.chat_message = "업로드하신 얼굴 사진에 해당하는 분의 성함을 입력해 주세요."
        st.session_state.json_saved = False

        # 업로드된 이미지를 사이드바에 표시
        if "image_uploaded" in st.session_state and st.session_state.image_uploaded:
            st.image(st.session_state.image, caption="업로드된 이미지", use_column_width=True)
        
###################################################################################################################
# 피부 분석 단계 (얼굴 부위 탐지, 분석결과, 요약) #
################################################
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'name_entered' not in st.session_state:
    st.session_state.name_entered = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'analysis_completed' not in st.session_state:  # 분석이 완료되었는지 확인하는 상태 변수
    st.session_state.analysis_completed = False
    
# 이전 채팅 내용 화면에 표기 (히스토리 유지)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("채팅을 입력해주세요.")    

# 피부 분석이 완료되지 않은 경우 -> 이름 입력 및 피부 분석 처리
if not st.session_state.analysis_completed:
    if "image_uploaded" in st.session_state and st.session_state.image_uploaded:
        st.chat_message("assistant").write(st.session_state.chat_message)
        st.empty()
        # 사용자 이름 입력 처리 
        if not st.session_state.name_entered:
            if user_input:
                st.session_state.name_entered = True
                st.session_state.user_name = user_input
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)
                
#######################################################################################################################
# 사용자 이름 입력 후 분석 시작 #
###############################                
                img_resized = ImageOps.fit(image, (416, 416), Image.Resampling.LANCZOS)
                img_rgb = img_resized.convert("RGB")  # 이미지를 PIL RGB로 변환
                image_rgb = np.array(img_rgb)  # YOLO 모델을 위한 NumPy 배열로 변환 (OpenCV 사용하지 않음)
                results = yolo_model(image_rgb)
                    
                # 분석 결과가 없는 경우
                if len(results.pandas().xyxy[0]) == 0:
                    st.write("이미지에서 얼굴을 식별하지 못했습니다. 다시 시도해 주세요.")
                    st.session_state.image_uploaded = False  # 이미지 분석 실패 시 상태를 초기화               
                    st.stop() 
                else:
                    pass

#######################################################################################################################
# 부위별 분석 #
##############
                with st.spinner('피부를 분석하는 중입니다. 잠시만 기다려주세요'):

                    # 메인 분석 로직
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    all_results = []
                    for annotation, class_id in model_class_id_map.items():
                        model_settings_for_annotation = model_settings[annotation]
                        resnet_model = DenseNet201_VGG19_Ensemble(
                            num_classes=model_settings_for_annotation['num_classes'],
                            drop_out=model_settings_for_annotation['dropout_rate']
                        )

                        state_dict = torch.load(model_settings_for_annotation['path'], map_location=device)
                        renamed_state_dict = rename_state_dict_keys(state_dict)
                        resnet_model.load_state_dict(renamed_state_dict)
                        resnet_model.to(device).eval()

                        # 크롭된 이미지 저장 및 결과 수집
                        results_for_annotation = cropped_img(annotation, resnet_model, results, class_id, img_rgb)
                        all_results.extend(results_for_annotation)

                    # JSON 파일로 저장
                    if all_results:
                        json_file_path = 'detection_results.json'
                        with open(json_file_path, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        st.session_state.json_data = all_results
                        st.session_state.json_saved = True

                        # # 분석 결과를 message_list에 추가
                        # st.session_state.chat_history.append({"role": "assistant", "content": "피부 분석 결과: " + str(all_results)})

#######################################################################################################################
# 관리가 필요한 부위 선별 #
#########################
                  
                detailed_result = create_prompt(st.session_state.json_data)    
#######################################################################################################################
# 결과 출력 #
############
                user_name = st.session_state.get('user_name', '사용자')
                display_results_and_summary(detailed_result, user_name)
                
                st.session_state.analysis_completed = True
                
#######################################################################################################################    
# 분석이 완료된 경우 -> 새로운 대화 처리
elif st.session_state.analysis_completed:
    
    if user_input:
        # 사용자 메시지 히스토리에 추가 및 화면에 출력
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Spinner가 동작 중이라는 상태를 플래그로 설정
        st.session_state.spinner_active = True

        # 빈 공간 생성 (Spinner가 돌아가는 동안 AI 응답이 비워지도록 함)
        response_placeholder = st.empty()

        # AI 응답 생성 및 출력
        with st.spinner("답변을 생성하는 중입니다~"):
            ai_response = run_retriever_agent(user_input)

        # Spinner 종료 후 메시지 출력
        st.session_state.spinner_active = False  # Spinner가 종료됨을 표시
        st.session_state.chat_history.append({"role": "ai", "content": ai_response})
        # Spinner가 끝난 후 AI 응답을 빈 공간에 표시
        with response_placeholder.chat_message("ai"):
            response_placeholder.write(ai_response)