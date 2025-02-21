import ast
import logging                   
import re                        
import traceback                 
from deep_translator import GoogleTranslator  
from dotenv import load_dotenv   
import streamlit as st           
import os                        
from datetime import datetime    
import pytz                      
import json                      
import uuid                      
import shutil                    
from supervisor import SupervisorAgent, PLOT_DIR  
from datahandle import DataHandler   
import matplotlib.pyplot as plt  
import numpy as np               

# โหลด environment variables จากไฟล์ .env
load_dotenv()

# กำหนดค่าคงที่และการตั้งค่าต่าง ๆ สำหรับแอปพลิเคชัน
APP_NAME = "(DEMO) Data Analysis Assistant 📊"  # ชื่อของแอปพลิเคชัน
BASE_SESSION_DIR = "sessions"            # โฟลเดอร์หลักสำหรับเก็บข้อมูล session ของผู้ใช้
TEMP_UPLOAD_DIR = "temp_uploads"         # โฟลเดอร์ชั่วคราวสำหรับเก็บไฟล์ที่อัปโหลดเข้ามา

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(
    page_title=APP_NAME,         # ชื่อของหน้าเว็บ
    page_icon="🐘",              # ไอคอนของหน้าเว็บ
    layout="wide",               # รูปแบบการแสดงผล (wide สำหรับแสดงเต็มหน้าจอ)
    initial_sidebar_state="expanded"  # ให้ sidebar แสดงผลโดยเริ่มต้น
)

# กำหนด Custom CSS เพื่อปรับแต่งรูปแบบการแสดงผลของหน้าเว็บ
st.markdown("""
<style>
    /* ปรับพื้นหลังของ container หลัก */
    .main {
        background-color: #343541;
        padding-bottom: 80px;
    }
    /* รูปแบบข้อความของผู้ใช้ */
    .user-message {
        background-color: #40414f;
        color: #d1d5db;
        padding: 15px 20px;
        margin: 20px 0;
        border-radius: 8px;
        max-width: 85%;
        align-self: flex-end;
    }
    /* รูปแบบข้อความของผู้ช่วย (Assistant) */
    .assistant-message {
        color: #d1d5db;
        border-radius: 8px;
        max-width: 85%;
        align-self: flex-start;
    }
    /* รูปแบบสำหรับแสดงข้อมูล session */
    .session-info {
        background-color: #2e7d32;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    /* ตำแหน่งและลักษณะของ container สำหรับ input ที่อยู่ด้านล่าง */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        margin-top: 20px;
        width: 100%;
        background-color: #343541;
        padding: 10px;
        border-top: 1px solid #565869;
    }
    /* รูปแบบของ input field สำหรับข้อความ */
    .stTextInput input {
        background-color: #40414f;
        color: white;
        border: 1px solid #565869;
        border-radius: 5px;
        width: 100%;
        padding: 10px;
    }
    /* รูปแบบของปุ่ม */
    .stButton button {
        background-color: #565869;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #6e7087;
    }
    /* รูปแบบ container สำหรับแสดงกราฟ */
    .plot-container {
        width: 100%;
        max-width: 800px;
        margin: 20px auto;
        padding: 10px;
        background-color: #2d2d3a;
        border-radius: 8px;
    }
    /* ปรับแต่ง Sidebar */
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    /* รูปแบบข้อความสถานะ (success, warning, error) */
    .success-message {
        background-color: #1a472a;
        color: #2ecc71;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .warning-message {
        background-color: #7d4a00;
        color: #f39c12;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .error-message {
        background-color: #700000;
        color: #e74c3c;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ฟังก์ชันสำหรับแปลง JSON (dict หรือ list) ให้เป็นข้อความที่มีการเยื้อง (indent) เพื่อให้อ่านง่าย
def convert_json_to_text(data, indent=0):
    """
    แปลงข้อมูล JSON (dict หรือ list) ให้เป็นข้อความที่อ่านง่าย
    Parameters:
        data: ข้อมูล JSON (dict หรือ list)
        indent: ระดับการเยื้อง (indentation) สำหรับการจัดรูปแบบข้อความ
    Returns:
        ข้อความที่ถูกจัดรูปแบบแล้ว
    """
    spaces = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            # หาก value เป็น dict หรือ list ให้เรียกฟังก์ชันนี้แบบ recursive เพิ่มการเยื้อง
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}{key.capitalize()}:")
                lines.append(convert_json_to_text(value, indent=indent + 2))
            else:
                lines.append(f"{spaces}{key.capitalize()}: {value}")
        return "\n".join(lines)
    elif isinstance(data, list):
        lines = []
        for index, item in enumerate(data):
            lines.append(f"{spaces}- {convert_json_to_text(item, indent=indent + 2)}")
        return "\n".join(lines)
    else:
        return f"{spaces}{data}"

# ฟังก์ชันสำหรับแปลง JSON (ไม่ว่าจะเป็นสตริงหรือ dict) ให้เป็นข้อความที่อ่านง่าย
def json_to_text(json_input):
    """
    รับ input เป็น JSON string หรือ dict แล้วแปลงเป็นข้อความที่อ่านง่าย
    Parameters:
        json_input: JSON string หรือ dict
    Returns:
        ข้อความที่อ่านง่าย
    """
    # หาก input เป็นสตริง ให้พยายามแปลงเป็น dict ก่อน
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError:
            # หากแปลงไม่ได้ให้คืนค่า input เดิม
            return json_input
    else:
        data = json_input

    return convert_json_to_text(data)

# ฟังก์ชันสำหรับแปลข้อความไปยังภาษาที่ต้องการโดยใช้ GoogleTranslator
def translate_func(target_lang, text):
    # แปลงข้อความ JSON ให้เป็นข้อความที่อ่านง่ายก่อน
    normal_text = json_to_text(text)
    max_char = 5000
    if len(normal_text) <= max_char:
        try:
            # แปลข้อความจากภาษาอัตโนมัติเป็นภาษาที่ target_lang ระบุไว้
            translated = GoogleTranslator(source='auto', target=target_lang).translate(normal_text)
            return translated
        except:
            # หากแปลไม่สำเร็จ คืนค่าข้อความต้นฉบับ
            return normal_text
    else:
        return normal_text

# ฟังก์ชันสำหรับรับ API key สำหรับ supervisor (LLM หลัก) ตามโมเดลที่เลือก
def get_supervisor_api_key(model):
    if model == "typhoon-v1.5x-70b-instruct":
        return os.getenv("TYPHOON_API_KEY")
    elif model == "typhoon-v2-70b-instruct":
        return os.getenv("TYPHOON_API_KEY")
    elif model == "gpt-4o-mini":
        return os.getenv("OPENAI_API_KEY")
    else:
        return None

# ฟังก์ชันสำหรับรับ API key สำหรับ agent (เช่น PandasAgent) ตามโมเดลที่เลือก
def get_agent_api_key(model):
    if model == "typhoon-v1.5x-70b-instruct":
        return os.getenv("PANDAS_API_KEY")
    elif model == "typhoon-v2-70b-instruct":
        return os.getenv("PANDAS_API_KEY")
    elif model == "gpt-4o-mini":
        return os.getenv("OPENAI_API_KEY")
    else:
        return None
    
# ฟังก์ชันสำหรับรับ API key สำหรับ explanner (LLM ย่อยให้คำอธิบาย) ตามโมเดลที่เลือก
def get_explanne_tool_api_key(model):
    if model == "typhoon-v1.5x-70b-instruct":
        return os.getenv("EXPLANNER_API_KEY")
    elif model == "typhoon-v2-70b-instruct":
        return os.getenv("EXPLANNER_API_KEY")
    elif model == "gpt-4o-mini":
        return os.getenv("OPENAI_API_KEY")
    else:
        return None

# ฟังก์ชันสำหรับรับ base URL สำหรับ API ตามโมเดลที่เลือก
def get_model_base_url(model):
    if model == "typhoon-v1.5x-70b-instruct":
        return "https://api.opentyphoon.ai/v1"
    elif model == "typhoon-v2-70b-instruct":
        return "https://api.opentyphoon.ai/v1"
    elif model == "gpt-4o-mini": # ดูเอกสาร: https://www.restack.io/p/openai-python-answer-base-url-cat-ai
        return "https://api.openai.com/v1"
    else: 
        return "https://api.opentyphoon.ai/v1"

# =======================================================================
# คลาส Session สำหรับเก็บข้อมูลการสนทนาและการจัดการ session
# =======================================================================
class Session:
    def __init__(self, session_id=None):
        # กำหนด session_id ถ้าไม่ระบุจะสร้างใหม่โดยใช้ uuid
        self.session_id = session_id or str(uuid.uuid4())
        # เก็บเวลาที่ session ถูกสร้างในรูปแบบ UTC
        self.created_at = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
        # รายการข้อความใน session (ทั้งจากผู้ใช้และผู้ช่วย)
        self.messages = []
        # เก็บไฟล์ที่ถูกอัปโหลด (ถ้ามี)
        self.uploaded_file = None
        # เก็บเส้นทางของไฟล์ที่ถูกอัปโหลด
        self.file_path = None
        # เก็บเวลาที่มีการใช้งาน session ครั้งสุดท้าย
        self.last_activity = self.created_at
        
    def to_dict(self):
        """
        แปลงข้อมูลของ session เป็น dict สำหรับการบันทึกลงไฟล์ JSON
        """
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'messages': self.messages,
            'file_path': self.file_path,
            'last_activity': self.last_activity
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        สร้าง instance ของ Session จาก dict ที่ได้บันทึกไว้
        """
        session = cls(session_id=data['session_id'])
        session.created_at = data['created_at']
        session.messages = data['messages']
        session.file_path = data['file_path']
        session.last_activity = data.get('last_activity', datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'))
        return session

    def update_activity(self):
        """
        อัปเดตเวลาที่มีการใช้งาน session
        """
        self.last_activity = datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')

# =======================================================================
# คลาส SessionManager สำหรับจัดการ session ทั้งหมด (สร้าง, โหลด, ลบ, รายการ)
# =======================================================================
class SessionManager:
    def __init__(self, base_dir=BASE_SESSION_DIR):
        # กำหนดโฟลเดอร์หลักสำหรับเก็บ session
        self.base_dir = base_dir
        # ตรวจสอบให้แน่ใจว่าโฟลเดอร์หลักมีอยู่ ถ้าไม่มีให้สร้างใหม่
        self.ensure_base_dir()
        
    def ensure_base_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)
        
    def get_session_dir(self, session_id):
        # คืนค่าเส้นทางของ session ตาม session_id
        return os.path.join(self.base_dir, session_id)
    
    def create_session(self):
        # สร้าง session ใหม่และบันทึกลงในระบบ
        session = Session()
        session_dir = self.get_session_dir(session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.save_session(session)
        return session
    
    def save_session(self, session):
        # บันทึกข้อมูล session ลงในไฟล์ session.json ภายในโฟลเดอร์ของ session
        session_dir = self.get_session_dir(session.session_id)
        os.makedirs(session_dir, exist_ok=True)
        session_file = os.path.join(session_dir, 'session.json')
        with open(session_file, 'w') as f:
            json.dump(session.to_dict(), f)
    
    def load_session(self, session_id):
        # โหลด session จากไฟล์ session.json ตาม session_id ที่ระบุ
        session_dir = self.get_session_dir(session_id)
        session_file = os.path.join(session_dir, 'session.json')
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                try:
                    data = json.load(f)
                    return Session.from_dict(data)
                except json.JSONDecodeError:
                    return None
        return None
    
    def delete_session(self, session_id):
        # ลบโฟลเดอร์ของ session ที่ระบุ (ลบไฟล์ทั้งหมดภายใน session นั้น)
        session_dir = self.get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
    
    def list_sessions(self):
        # คืนค่าเป็นรายการ session ทั้งหมดที่มีอยู่ โดยเรียงลำดับตามเวลาที่มีการใช้งานล่าสุด (ล่าสุดมาก่อน)
        if not os.path.exists(self.base_dir):
            return []
        sessions = []
        for session_id in os.listdir(self.base_dir):
            session = self.load_session(session_id)
            if session:
                sessions.append(session)
        return sorted(sessions, key=lambda x: x.last_activity, reverse=True)

# =======================================================================
# ฟังก์ชันสำหรับจัดการไฟล์ที่ผู้ใช้อัปโหลด (บันทึกและลบไฟล์ที่เกี่ยวข้องกับ session)
# =======================================================================
def save_uploaded_file(uploaded_file, session_id):
    """
    บันทึกไฟล์ที่ผู้ใช้อัปโหลดลงในโฟลเดอร์ของ session
    Parameters:
        uploaded_file: ไฟล์ที่อัปโหลด (Streamlit UploadedFile)
        session_id: ID ของ session ปัจจุบัน
    Returns:
        เส้นทางของไฟล์ที่ถูกบันทึก หากไม่มีไฟล์ให้คืนค่า None
    """
    if uploaded_file is not None:
        session_dir = os.path.join(BASE_SESSION_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        file_path = os.path.join(session_dir, uploaded_file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logging.info(f"File saved successfully at {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            st.error(f"Error saving file: {e}")
            return None
    return None


def delete_session_file(session_id):
    """
    ลบไฟล์ทั้งหมดภายในโฟลเดอร์ของ session ยกเว้นไฟล์ session.json
    Parameters:
        session_id: ID ของ session ที่ต้องการลบไฟล์
    """
    session_dir = os.path.join(BASE_SESSION_DIR, session_id)
    if os.path.exists(session_dir):
        for file in os.listdir(session_dir):
            if file != 'session.json':
                os.remove(os.path.join(session_dir, file))

# =======================================================================
# ฟังก์ชันสำหรับโหลดข้อมูลจากไฟล์ที่ถูกอัปโหลด
# ใช้ DataHandler ในการโหลดและ preprocess ข้อมูล
# =======================================================================

@st.cache_data
def load_data(file_path, session_id):
    if not file_path:
        st.error("No file path provided.")
        return None

    data_handler = DataHandler({})
    file_key = os.path.splitext(os.path.basename(file_path))[0]
    data_handler.dataset_paths[file_key] = file_path
    try:
        data_handler.load_data()
        data_handler.preprocess_data()
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}")
        logging.error(f"Error loading file: {e}")
        return None
    return data_handler
# =======================================================================
# Initializations: กำหนดค่าเริ่มต้นใน session state ของ Streamlit
# =======================================================================
if 'session_manager' not in st.session_state:
    st.session_state['session_manager'] = SessionManager()
if 'current_session' not in st.session_state:
    st.session_state['current_session'] = None
if 'data_handler' not in st.session_state:
    st.session_state['data_handler'] = DataHandler({})
if 'supervisor_agent' not in st.session_state:
    st.session_state['supervisor_agent'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'initial_message_sent' not in st.session_state:
    st.session_state['initial_message_sent'] = False 

# =======================================================================
# ฟังก์ชันสำหรับจัดการ session (เริ่ม session ใหม่, เปลี่ยน session, ลบ session, ล้างประวัติการสนทนา)
# =======================================================================
def start_new_session():
    """
    สร้าง session ใหม่และบันทึกลงในระบบ พร้อมแสดงข้อความแจ้งเตือน
    Returns:
        instance ของ Session ที่สร้างขึ้นใหม่
    """
    session = st.session_state['session_manager'].create_session()
    st.session_state['current_session'] = session
    st.success(f"Started new session: {session.session_id[:8]}")
    return session

def switch_session(session_id: str, selected_model: str, temperature: float):
    """
    เปลี่ยนไปใช้ session ที่ระบุและโหลดข้อมูลที่เกี่ยวข้อง พร้อมปรับค่า SupervisorAgent ตามโมเดลและ temperature ที่เลือก
    """
    try:
        # โหลด session จากระบบ
        session = st.session_state['session_manager'].load_session(session_id)
        if not session:
            raise ValueError(f"Session with ID {session_id} not found.")

        # ตั้งค่า session ปัจจุบัน
        st.session_state['current_session'] = session
        
        # ตรวจสอบว่ามีไฟล์ใน session หรือไม่
        if session.file_path and os.path.exists(session.file_path):
            # โหลดข้อมูลจากไฟล์ของ session นั้น
            st.session_state['data_handler'] = load_data(session.file_path, session.session_id)
            
            # กำหนด dataset key จากชื่อไฟล์ที่อัปโหลด
            dataset_key = os.path.splitext(os.path.basename(session.file_path))[0]
            
            # สร้าง instance ของ SupervisorAgent
            st.session_state['supervisor_agent'] = SupervisorAgent(
                temperature=temperature,
                base_url=get_model_base_url(selected_model),
                model_name=selected_model,
                dataset_paths={dataset_key: session.file_path},
                dataset_key=dataset_key,
                session_id=session.session_id,
                supervisor_api_key=get_supervisor_api_key(selected_model),
                agent_api_key=get_agent_api_key(selected_model),
                explanner_api_key=get_explanne_tool_api_key(selected_model),
            )

            logging.info(f"Switched to session {session_id} with dataset {dataset_key}")
        else:
            # หากไม่มีไฟล์ ให้รีเซ็ต data handler และ agent
            st.session_state['data_handler'] = DataHandler({})
            st.session_state['supervisor_agent'] = None
            st.warning("⚠️ This session has no dataset. Please upload a file to continue.")

    except Exception as e:
        logging.error(f"Error switching session: {str(e)}")
        logging.error(traceback.format_exc())
        st.error(f"Error switching session: {str(e)}")


def delete_current_session():
    """
    ลบ session ปัจจุบัน พร้อมล้างข้อมูลที่เกี่ยวข้องใน session state
    """
    if st.session_state['current_session']:
        session_id = st.session_state['current_session'].session_id
        st.session_state['session_manager'].delete_session(session_id)
        st.session_state['current_session'] = None
        st.session_state['supervisor_agent'] = None
        st.session_state['data_handler'] = DataHandler({})
        st.success(f"Session {session_id[:8]} deleted")

def clear_chat_history():
    """
    ล้างประวัติการสนทนาใน session ปัจจุบันและบันทึกการเปลี่ยนแปลง
    """
    if st.session_state['current_session']:
        st.session_state['current_session'].messages = []
        st.session_state['session_manager'].save_session(st.session_state['current_session'])
        if st.session_state['supervisor_agent']:
            st.session_state['supervisor_agent'].clear_memory()
        st.success("Chat history cleared")

# =======================================================================
# ฟังก์ชันสำหรับจัดการการส่งข้อความจากผู้ใช้
# =======================================================================
def handle_submit(user_input):
    """
    จัดการการส่งข้อความจากผู้ใช้:
      - ตรวจสอบว่ามี session ปัจจุบันหรือไม่
      - เพิ่มข้อความของผู้ใช้เข้าไปในประวัติการสนทนา
      - เรียกใช้งาน SupervisorAgent เพื่อรับคำตอบ
      - เพิ่มคำตอบจากผู้ช่วยลงในประวัติและบันทึก session
      - รันสคริปต์ใหม่เพื่ออัปเดตหน้าจอ
    Parameters:
        user_input: ข้อความที่ผู้ใช้ป้อนเข้ามา
    """
    if not st.session_state['current_session']:
        st.warning("Please start or select a session first")
        return
        
    if user_input.strip() and st.session_state['supervisor_agent']:
        current_session = st.session_state['current_session']
        current_session.update_activity()
        
        # เพิ่มข้อความจากผู้ใช้ลงใน session
        message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
        }
        current_session.messages.append(message)
        st.session_state['messages'].append(message)
        
        try:
            with st.spinner("🤖 Assistant is typing..."):
                # ส่งข้อความไปยัง SupervisorAgent และรับคำตอบ
                response = st.session_state['supervisor_agent'].run(user_input)
                
                # เพิ่มข้อความจากผู้ช่วยลงใน session
                message = {
                    "role": "assistant",
                    "content": response.model_dump(),
                    "timestamp": datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
                }
                current_session.messages.append(message)
                st.session_state['messages'].append(message)
                
                # บันทึก session ปัจจุบัน
                st.session_state['session_manager'].save_session(current_session)
                
                # Log the response for debugging
                logging.info(f"Response from SupervisorAgent: {response.model_dump()}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logging.error(f"Error in handle_submit: {str(e)}")
            logging.error(traceback.format_exc())
        
        # รันสคริปต์ใหม่เพื่ออัปเดตหน้าจอ (รีเฟรช UI)
        st.rerun()

# =======================================================================
# ฟังก์ชัน main สำหรับจัดการส่วน UI ของแอปพลิเคชัน
# =======================================================================
def main():
    # แสดงชื่อแอปพลิเคชันบนหน้าเว็บ
    st.title(APP_NAME)
    
    # แสดงส่วน Console logs ภายใน expander (สำหรับ debug)
    with st.expander("Thought logs.", expanded=False):
        if st.session_state['current_session']:
            for message in st.session_state['current_session'].messages:
                # หากเป็นข้อความจาก assistant ที่มีข้อมูล raw_response ให้จัดรูปแบบและแสดง
                if message["role"] == "assistant" and "raw_response" in message["content"]:
                    raw_response = message["content"]["raw_response"]
                    if (raw_response):
                        formatted_text = re.sub(r"(Thought:|Final Answer:|Action:|Action Input:|Observation:|Action Output:)", r"\n\1", raw_response)
                        formatted_text = re.sub(r"<br>\s*<br>", "<br>", formatted_text)
                        st.markdown(formatted_text, unsafe_allow_html=True)
                    else:
                        st.write("No console logs available.")
        else:
            st.write("📜 All logs.")
    # เพิ่มส่วนแสดงตารางข้อมูล (Data Preview) ด้านล่าง Console logs
    if st.session_state.get('current_session') and st.session_state['current_session'].file_path:
        dataset_key = os.path.splitext(os.path.basename(st.session_state['current_session'].file_path))[0]
        try:
            # ดึง DataFrame จาก DataHandler โดยใช้ dataset key
            df = st.session_state['data_handler'].get_data(dataset_key)
            st.subheader("Data Preview")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error loading data table: {e}")
    
    # Sidebar สำหรับตั้งค่าต่าง ๆ
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # แสดงเวลาปัจจุบันในรูปแบบ UTC
        st.info(f"🕒 UTC: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # แสดงข้อมูล session ปัจจุบัน (ถ้ามี)
        current_session = st.session_state.get('current_session')
        if current_session:
            st.subheader("Current Session")
            st.info(f"Session ID: {current_session.session_id[:8]}")
        
        # ส่วนสำหรับจัดการ session (สร้าง session ใหม่, เปลี่ยน session, ลบ session)
        st.header("Session Management")
        if st.button("🆕 New Chat Session"):
            start_new_session()

        # ตัวเลือกสำหรับเลือกโมเดลที่ต้องการใช้งาน
        # model_options = {
        #     "typhoon-v1.5x": "typhoon-v1.5x-70b-instruct",
        #     "typhoon-v2": "typhoon-v2-70b-instruct",
        #     "open_ai": "gpt-4o-mini"
        # }

        model_options = {
            "typhoon-v1.5x": "typhoon-v1.5x-70b-instruct",
            "typhoon-v2": "typhoon-v2-70b-instruct",
        }
        selected_model: str = st.selectbox("🦾 Model Settings", list(model_options.values()))
        
        # เลือกค่า temperature สำหรับโมเดล (ความสุ่มในการตอบ)
        temperature: float = st.select_slider(
            'Set temperature',
            options=[round(i * 0.1, 1) for i in range(0, 11)],
            value=0.3
        )
        
        # แสดงรายการ session ทั้งหมดที่มีอยู่ในระบบ
        sessions = st.session_state['session_manager'].list_sessions()
        if sessions:
            st.subheader("Your Sessions")
            for session in sessions:
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        # เพิ่มไอคอนแสดงสถานะไฟล์
                        file_status = "📁" if session.file_path else "📭"
                        st.info(f"{file_status} Session {session.session_id[:8]}\n{session.last_activity}")            
                    with col2:
                        if st.button("Switch", key=f"switch_{session.session_id}"):
                            switch_session(session.session_id, selected_model, temperature)
                    with col3:
                        if st.button("Delete", key=f"delete_{session.session_id}"):
                            if st.session_state['current_session'] and session.session_id == st.session_state['current_session'].session_id:
                                delete_current_session()
                            else:
                                st.session_state['session_manager'].delete_session(session.session_id)
        
        # ส่วนจัดการไฟล์สำหรับ session ปัจจุบัน
        if st.session_state['current_session']:
            st.header("File Management")
            current_session = st.session_state['current_session']

            # หากยังไม่มีไฟล์ที่อัปโหลดใน session ให้แสดง file uploader
            if not current_session.file_path:
                st.warning("""
                ⚠️ Warning !!
                ### 📤 Upload Your Dataset
                
                To get started, please upload your data file. We support:
                - CSV files (.csv)
                - Excel files (.xls, .xlsx)
                
                Your data will be securely stored and only accessible within this session.
                """)
                
                uploaded_file = st.file_uploader(
                    "📂 Upload Dataset",
                    type=['csv', 'xls', 'xlsx'],
                    key='file_uploader',
                    help="Select your dataset file to begin."
                )

                if uploaded_file:
                    file_path = save_uploaded_file(uploaded_file, current_session.session_id)
                    st.session_state['initial_message_sent'] = False
                    if file_path:
                        try:
                            data_handler = load_data(file_path, current_session.session_id)
                            st.session_state['data_handler'] = data_handler
                            current_session.file_path = file_path
                            
                            # Initialize SupervisorAgent ด้วยข้อมูลไฟล์ที่อัปโหลด
                            dataset_key = os.path.splitext(uploaded_file.name)[0]
                            st.session_state['supervisor_agent'] = SupervisorAgent(
                                temperature=temperature,
                                base_url=get_model_base_url(selected_model),
                                model_name=selected_model,
                                dataset_paths={dataset_key: file_path},
                                dataset_key=dataset_key,
                                session_id=current_session.session_id, 
                                supervisor_api_key=get_supervisor_api_key(selected_model),
                                agent_api_key=get_agent_api_key(selected_model),
                                explanner_api_key=get_explanne_tool_api_key(selected_model),
                            )

                            st.session_state['session_manager'].save_session(current_session)
                            st.success(f"Successfully loaded {uploaded_file.name}")
                            # ส่งข้อความเริ่มต้นให้ Assistant หากยังไม่ได้ส่ง
                            if not st.session_state['initial_message_sent']:
                                handle_submit(user_input="🔛 start Assistant system.")
                                st.session_state['initial_message_sent'] = True
                        except Exception as e:
                            st.error(f"Error loading file: {str(e)}")
            else:
                st.info(f"Current file: {os.path.basename(current_session.file_path)}")

    # =======================================================================
    # ส่วนของหน้าจอ Chat Interface (การแสดงผลข้อความและ input สำหรับแชท)
    
    # แสดงข้อความแชททั้งหมดใน session
    if st.session_state['current_session']:
        for message in st.session_state['current_session'].messages:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div style="display: flex; justify-content: space-between;">
                            <div>{message["content"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if "sub_response" in message["content"]:
                        sub_resp = message["content"]["sub_response"]
                        # แสดงผลจาก pandas_agent (เดิม)
                        if "pandas_agent" in sub_resp:
                            pandas_response = sub_resp["pandas_agent"]
                            if pandas_response.get("execution_result"):
                                if "plots" in pandas_response["execution_result"] and pandas_response["execution_result"]["plots"]:
                                    for plot in pandas_response["execution_result"]["plots"]:
                                        with st.container():
                                            st.markdown("🐼 Assistant (Pandas Agent):")
                                            st.image(os.path.join("static", "plots", plot["filename"]), width=800)
                                            with st.expander("Show plot details"):
                                                st.markdown('</div>', unsafe_allow_html=True)
                                                st.code(pandas_response["execution_result"]["output"])    
                                elif "output" in pandas_response["execution_result"]:
                                    st.markdown("🐼 Assistant (Pandas Agent):")
                                    st.code(pandas_response["execution_result"]["output"])
                            if pandas_response.get("code"):
                                with st.expander("Show Code from Pandas Agent"):
                                    st.code(pandas_response["code"], language="python")
                            if pandas_response.get("explanation"):
                                st.write("🐼 Assistant (Pandas Agent):")
                                explan = pandas_response["explanation"].get("explanation", "")
                                try:
                                    st.write(translate_func(target_lang='th', text=explan))
                                except Exception:
                                    st.write(explan)
                    
                        # ส่วนแสดงผลสำหรับ analysis_agent
                        elif "analysis_agent" in sub_resp:
                            analysis_response = sub_resp["analysis_agent"]
                            
                            # แสดงส่วนหัวของ Assistant
                            st.markdown("""
                            <div class="assistant-message">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>🧑🏻‍🏫 Assistant (Analysis Agent):</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # แสดงผลการวิเคราะห์
                            with st.expander("🔍 Analysis Details", expanded=True):
                                # 1. แสดงคำอธิบาย
                                if "explanation" in analysis_response:
                                    explanation = analysis_response["explanation"]
                                    if isinstance(explanation, dict):
                                        # st.markdown(f"**คำอธิบาย:** {explanation.get('text', '')}")
                                        st.markdown(f"**คำอธิบาย:** {translate_func(target_lang='th', text=explanation.get('text', ''))}")
                                    else:
                                        st.markdown(f"**คำอธิบาย:** {explanation}")
                                
                                # 2. แสดงโค้ด (ถ้ามี)
                                if analysis_response.get("code"):
                                    with st.expander("🔍 แสดงโค้ดที่ใช้ในการวิเคราะห์"):
                                        st.code(analysis_response["code"], language="python")
                                
                                # 3. แสดงผลลัพธ์การรัน (ถ้ามี)
                                if analysis_response.get("execution_result"):
                                    with st.expander("📊 ผลลัพธ์การวิเคราะห์"):
                                        st.write(analysis_response["execution_result"])
                                
                                # 4. แสดง response (ถ้ามี)
                                if analysis_response.get("response"):
                                    st.markdown("**ผลการวิเคราะห์:**")
                                    st.write(analysis_response["response"])


                        # แสดงข้อความตอบกลับหลัก (ถ้ามี)
                        if message["content"].get("response", ""):
                            st.markdown(f"""
                            <div class="assistant-message">
                                <div style="display: flex; justify-content: space-between;">
                                    <div>🤖 Assistant: {message["content"].get("response", "")}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="assistant-message">
                            <div style="display: flex; justify-content: space-between;">
                                <div>🤖 Assistant: {message["content"].get("response", "")}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


        # ช่องสำหรับรับข้อความจากผู้ใช้ (chat input)
        user_input = st.chat_input(
            key='user_input',
            placeholder="Type your message and press Enter"
        )
        if user_input:
            handle_submit(user_input)
    else:
        st.warning("""
            ⚠️ Warning !! \n 
            Dataset Required

            To start using this session, please upload a dataset file. 
            Supported file types: CSV, Excel (XLS/XLSX)

            1. Click the 'Upload Dataset' button on sidebar
            2. Select your file
            3. Wait for the file to be processed

            Once uploaded, you can start analyzing your data!
            """)
        st.info("Please start a new chat session or select an existing one.")

# =======================================================================
# เรียกใช้ฟังก์ชัน main เมื่อสคริปต์ถูกเรียกใช้งานโดยตรง
# =======================================================================
if __name__ == "__main__":
    main()