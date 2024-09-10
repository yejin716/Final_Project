################################라이브러리##########################################
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import random
import json
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from PIL import Image, ExifTags, ImageOps
import string
import torch
from langchain_core.runnables import(
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableMap
)
import base64
from langchain_core.messages import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from Final_utils import (print_messages, get_session_history, load_models, StreamHandler,
                    rename_state_dict_keys, cropped_img, create_prompt, display_results_and_summary,
                    classify_and_extract_keywords, studies_chain, prod_chain, general_chain, 
                    process_user_query,get_session_history)
from dotenv import load_dotenv
load_dotenv()


def show_home():
    # 이미지 파일을 base64로 변환하는 함수
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # 배경 이미지 설정
    background_img = get_img_as_base64(r"D:\MulCam\팀프로젝트\최종프로젝트0712_0910\Code\Code\LLM\Streamlit\entourage_main.png")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data::image/png;base64,{background_img}");
    background-size: cover;
    background-position: center -10%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    
    }}
    </style>
    """

    # 배경 이미지 적용
    st.markdown(page_bg_img, unsafe_allow_html=True) 
        
    
######################page 시작을 위한 초기 설정 #########################
def show_chatbot():
    st.markdown(
        """
        <style>
        .title-right { /* 여기에서 클래스 이름을 맞춤 */
            font-family: "Pretendard Variable", Pretendard; /* 글꼴 스타일 변경 */
            font-size: 48px; /* 글꼴 크기 */
            color: #2c3e50; /* 글자 색상 */
            font-weight: bold; /* 글자 두께 */
            margin-bottom: 0px; /* 아래 여백 */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="title-right">DERMA A.I</h1>', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: margin: 0; padding: 0;">──────────</h4>', unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* Google Fonts에서 Roboto 폰트 로드 */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        /* 커스텀 서브헤더 스타일 */
        .custom-subheader {
            font-size: 18px;  /* 원하는 글씨 크기로 설정 */
            font-family: 'Roboto', sans-serif;  /* 폰트 패밀리 설정 */
            color: #333;  /* 글씨 색상 설정 */
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="custom-subheader">인공지능 피부 분석을 통한 맞춤형 솔루션을 만나보세요!</p>', unsafe_allow_html=True)

    st.markdown("""
                <style>
                .custom-caption {
                    font-size: 14px; /* st.caption 기본 글씨 크기와 맞추기 */
                    color: #6c757d; /* st.caption 기본 색상과 맞추기 */
                    line-height: 1.5; /* 줄 간격 조절 */
                }
                </style>
                <p class="custom-caption">
                    지금 피부 사진을 업로드하시거나, 평소 피부에 대한 고민과 궁금증을 이야기해 주세요.<br>
                    맞춤형 솔루션으로 당신의 아름다움을 찾아드립니다.
                </p>
                """, unsafe_allow_html=True)
    
    ########################메시지 초기화 구간#####################

    # 초기 메시지 시작할 때에 message container 만들어 이곳에 앞으로 저장
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [] # 아예 내용을 지우고 싶다면 리스트 안의 내용을 clear 해주면 된다.

    # 채팅 대화기록을 저장하는 store 세션 상태 변수
    if 'store' not in st.session_state:
        st.session_state['store'] = dict()

    # 이미지 업로드 
    if 'image_uploaded' not in st.session_state:
        st.session_state['image_uploaded'] = []
    
    if 'is_logged_in' not in st.session_state:
        st.session_state['is_logged_in'] = False

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = None
        
    if 'yolo_model' not in st.session_state:
        yolo_model, model_class_id_map, model_settings, DenseNet201_VGG19_Ensemble, transform = load_models()
        st.session_state['yolo_model'] = yolo_model
        st.session_state['model_class_id_map'] = model_class_id_map
        st.session_state['model_settings'] = model_settings
        st.session_state['DenseNet201_VGG19_Ensemble'] = DenseNet201_VGG19_Ensemble
        st.session_state['transform'] = transform
    else:
        # 세션 상태에서 YOLO 모델 관련 변수 불러오기
        yolo_model = st.session_state['yolo_model']
        model_class_id_map = st.session_state['model_class_id_map']
        model_settings = st.session_state['model_settings']
        DenseNet201_VGG19_Ensemble = st.session_state['DenseNet201_VGG19_Ensemble']
        transform = st.session_state['transform']
        
    ################################## 사이드바 #########################
    def generate_random_id():
        # 닉네임 (id) 생성
        nicknames = ["꿀피부의역습","로션이랑나랑","미스트퀸카","에센스도둑","보습신공",
                    "모공잡는레인저","토너는사랑이야","피부계의수분폭격기","모찌피부짱",
                    "크림마녀의비밀","팩하는밤","피부천국행특급열차","세럼의유혹","각질요정등장",
                    "클렌징의여왕","샤이닝피부왕국","알로에마법사","수분을찾아서","피부의반란",
                    "밤새고촉촉"]
        return random.choice(nicknames)

    ### 이부분 부터 ###

    def reset_session_state():
        st.session_state['messages'] = []
        st.session_state['session_id'] = None  # ID 초기화
        st.session_state['is_logged_in'] = False
        st.session_state['image_uploaded'] = False
        
    with st.sidebar:
            
        def making_id():
            created_id = generate_random_id()
            st.session_state['session_id'] = created_id

        col1, col2 = st.columns(2)
        with col1:
            if st.button('닉네임 생성'):
                making_id()
            
        with col2:
            if st.button('로그아웃'):
                reset_session_state()
                st.rerun()
                
        session_id = st.text_input('Session ID', value=st.session_state['session_id'], key='session_id')
        
    ### 이 부분 까지 수정입니다 ###

        # 로그인 버튼을 눌렀을 때 session_id를 세션 상태에 저장
        if st.button('로그인'):
            if st.session_state.session_id:
                st.session_state['is_logged_in'] = True  # 로그인 상태 저장
                get_session_history(session_id)
            else:
                st.sidebar.write("Session ID를 입력하세요.")
            
            # 로그인 성공 후 세션 상태 확인
            if st.session_state.get('is_logged_in', False):
                st.sidebar.write(f"현재 로그인된 닉네임: {st.session_state['session_id']}")
            else:
                st.sidebar.write("로그인 필요")

        # 로그인된 상태에서 이미지 업로드 처리
        if st.session_state.get('is_logged_in', ""):
            st.sidebar.markdown('<h5 style="text-align: center;">이미지 업로드</h5>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.875rem;">✔<strong>메이크업을 하지 않은 사진</strong>을 올려주세요.</p>', unsafe_allow_html=True)
            st.markdown('<p style="font-size: 0.875rem;">✔<strong>깔끔한 배경</strong>에서 <strong>정면</strong>으로 찍으시면 더 정확하게 분석해드립니다.</p>', unsafe_allow_html=True)
            
            # 이미지 업로드
            file = st.sidebar.file_uploader('이미지를 업로드하세요', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

            if file is not None:
                st.session_state['uploaded_file'] = file  # 세션에 파일 저장
                image = Image.open(file)
                
                # 이미지 회전 정보 처리
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
                    pass

                # 업로드된 이미지 표시 및 버튼 출력

                # 업로드된 이미지 표시 및 버튼 출력
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("업로드된 이미지 보기"):
                        st.image(image, caption="업로드된 이미지", use_column_width=True)
                with col2:
                    image_btn = st.button('이미지 분석 시작')
                if image_btn:
                    st.session_state['image_uploaded'] = True
                
                    
            with st.expander("천천히 읽어보세요!", expanded=False):  # 기본적으로 접혀 있는 상태로 설정
                st.markdown('<p style="font-size: 14px;">✔ 조명이 어둡지 않은지 확인하세요.</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 14px;">✔ 앞머리는 넘기고 안경, 마스크, 모자는 벗어주세요.</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-size: 14px;">사진 촬영 환경에 따라 피부 상태가 다르게 나타날 수 있으므로, 최상의 분석 결과를 위해 위의 지침을 따라 주시기 바랍니다.</p>', unsafe_allow_html=True) 

        else:
            st.session_state['image_uploaded'] = False
            # st.session_state['session_id'] = False
    #############################################################################################################################################################################################################
    # 이미지 분석 모드
    try:
        ## 분석 시작 ##
        if st.session_state['image_uploaded']:
            # 세션 기록을 기반으로 ID 조회
            get_session_history(session_id)
            
            # 이전 분석 결과의 밑에 스피너를 고정할 위치를 지정
            spinner_placeholder = st.empty()
            with st.sidebar:
                with st.spinner('피부를 분석하는 중입니다. 잠시만 기다려주세요...'):
                    # 이미지 리사이즈 및 RGB 변환
                    img_resized = ImageOps.fit(image, (416, 416), Image.Resampling.LANCZOS)
                    img_rgb = img_resized.convert("RGB")
                    image_rgb = np.array(img_rgb)

                    # YOLO 모델로 이미지 분석 수행
                    results = yolo_model(image_rgb)

                    # 분석 결과가 없는 경우 처리
                    if len(results.pandas().xyxy[0]) == 0:
                        st.write("이미지에서 얼굴을 식별하지 못했습니다. 다시 시도해 주세요.")
                        st.session_state['image_uploaded'] = False
                        st.stop()
                    else:
                        # 분석 진행
                        pass

                    ## 부위별 분석 ##
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
                            json_file_path = f'./detection_results.json'
                            with open(json_file_path, 'w') as f:
                                json.dump(all_results, f, indent=2)

                            # 상세 분석 결과 생성 및 출력
                            detailed_result = create_prompt(all_results)
                
                ## 결과 출력 ##
                display_results_and_summary(detailed_result, session_id)

                st.session_state['image_uploaded'] = False
            
    except Exception as e:
        st.write(f"Error: {str(e)}")
    

    ############################# 이전 대화기록 누적/출력해 주는 코드 #############################
    # 이전 대화기로기을 출력해 주는 코드
    print_messages()

    #########################function Group##################################

    # 세션 ID를 기반으로 세션 기록을 가져오는 함수
    get_session_history(session_id)

    ############################실제 코드가 돌아가는 구간######################
    ############################사용자 메시지 입력구간###########################

    # 사용자 메시지 들어오는 곳
    if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
        # ID가 없을 때 안내 메시지
        st.chat_message('assistant').write("닉네임을 생성하고 로그인해주세요.")
    else:
        # Session ID가 있는 경우에만 메시지 입력 가능
        if user_input := st.chat_input('피부에 대한 고민을 들어드립니다. 무엇이든 물어보세요'):
            # 사용자가 입력한 내용을 출력
            st.chat_message('user').write(f'{user_input}') 
            st.session_state['messages'].append(ChatMessage(role='user', content=user_input))

            with st.spinner("답변을 생성하고 있습니다."):
                # AI의 답변 생성
                with st.chat_message('assistant'):

                    # 세션 기록을 기반으로 한 RunnableWithMessageHistory 설정
                    process_user_query_runnable = RunnableLambda(
                        lambda inputs: process_user_query(inputs["question"], inputs["session_id"])
                    )

                    # 실제 RunnableWithMessageHistory 가 적용된 Chain
                    with_message_history = RunnableWithMessageHistory(
                        process_user_query_runnable,
                        get_session_history,
                        input_messages_key="question",
                        history_messages_key="history",
                    )
                    
                    # 질문이 들어오면 실행 (chain 실행)
                    response = with_message_history.invoke(
                        {"question": user_input, "session_id": st.session_state['session_id']},  
                        config={"configurable": {"session_id": st.session_state['session_id']}}
                    )
                
                    
                    # 최종 invoke한 내용을 response에 넣었고 그것을 contents에 저장
                    st.session_state['messages'].append(ChatMessage(role='assistant', content=response))


def show_our_story():

    # 이미지 파일을 base64로 변환하는 함수
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # 배경 이미지 설정
    background_img = get_img_as_base64("./our_story.png")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data::image/png;base64,{background_img}");
    background-size: cover;
    background-position: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    </style>
    """

    # 배경 이미지 적용
    st.markdown(page_bg_img, unsafe_allow_html=True)
    






