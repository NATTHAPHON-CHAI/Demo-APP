import os
import sys      
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd            
import logging                 
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from tabulate import tabulate
from datahandle import DataHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json  
from prompt import get_prefix, get_suffix

# โหลด environment variables จากไฟล์ .env 
load_dotenv()

# ปิดการติดตามข้อมูลของ LangSmith โดยตั้งค่า environment variable ให้เป็น "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# =======================================================================
# กำหนดโมเดล Pydantic สำหรับโครงสร้างของ output ที่ agent จะส่งกลับมา
# =======================================================================
class PlotResponse(BaseModel):
    # Field สำหรับเก็บคำถามของผู้ใช้
    query: str = Field(description="User query")
    # Field สำหรับเก็บคำอธิบายอย่างละเอียดของการวิเคราะห์ที่ agent ทำขึ้น
    explanation: str = Field(description="Detailed explanation of the analysis")
    # Field สำหรับเก็บโค้ด Python ที่ agent สร้างขึ้นเพื่อให้ผู้ใช้สามารถนำไปใช้งานได้
    code: str = Field(description="The python code")

    class Config:
        extra = "forbid"


# =======================================================================
# PandasAgent สำหรับจัดการการทำงานเกี่ยวกับ DataFrame และสร้าง agent
# =======================================================================
class PandasAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, 
                 dataset_paths: dict, session_id: str, api_pandas_key: str):
        """
        ฟังก์ชันตัวสร้างสำหรับ PandasAgent
        Parameters:
            temperature (float): ค่าความสุ่ม (temperature) ที่ใช้กับโมเดลภาษา
            base_url (str): URL พื้นฐานสำหรับเข้าถึง API ของโมเดลภาษา
            model_name (str): ชื่อของโมเดลภาษา (เช่น gpt-3.5-turbo)
            dataset_paths (dict): เส้นทางไปยังชุดข้อมูลที่ต้องการใช้งาน
            session_id (str): รหัสประจำ session สำหรับติดตามการทำงาน
            api_pandas_key (str): API key สำหรับเข้าถึงโมเดลภาษาในส่วนของ PandasAgent
        """
        # สร้าง instance ของ DataHandler เพื่อจัดการการโหลดและ preprocess ข้อมูล
        self.handler = DataHandler(dataset_paths=dataset_paths)
        # โหลดข้อมูลจากชุดข้อมูลที่ระบุไว้ใน dataset_paths
        self.handler.load_data()
        # ทำการ preprocess ข้อมูล (เช่น การทำความสะอาดและแปลงรูปแบบข้อมูล)
        self.handler.preprocess_data()
        # เก็บค่าพารามิเตอร์ที่ได้รับมาไว้ใน attribute ของ instance
        self.temperature = temperature
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_pandas_key
        self.session_id = session_id
        # เริ่มต้นโมเดลภาษา (LLM) โดยเรียกใช้เมธอด initialize_llm()
        self.llm = self.initialize_llm()
        # สร้าง output parser โดยใช้ PydanticOutputParser พร้อมระบุโมเดล PlotResponse
        self.output_parser = PydanticOutputParser(pydantic_object=PlotResponse)

    def initialize_llm(self) -> ChatOpenAI:
        """
        ฟังก์ชันสำหรับเริ่มต้นโมเดลภาษา (LLM) ที่จะใช้สำหรับวิเคราะห์ข้อมูล
        Returns:
            instance ของ ChatOpenAI ที่ถูกกำหนดค่าแล้ว
        """
        # ตรวจสอบว่า API key ถูกระบุหรือไม่ หากไม่พบจะโยนข้อผิดพลาด
        if not self.api_key:
            raise ValueError("API key is missing. Ensure 'PANDAS_API_KEY' is set in your environment.")
        # สร้างและคืนค่า instance ของ ChatOpenAI ด้วยพารามิเตอร์ที่จำเป็น
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_agent(self, df_key: str):
        """
        ฟังก์ชันสำหรับสร้าง agent ที่สามารถทำงานกับ DataFrame ได้
        Parameters:
            df_key (str): คีย์ที่ระบุชุดข้อมูลที่ต้องการใช้งาน
        Returns:
            agent ที่ถูกสร้างขึ้นสำหรับวิเคราะห์ข้อมูลใน DataFrame
        """
        # ตรวจสอบว่าชุดข้อมูลที่ระบุมีอยู่ใน DataHandler หรือไม่
        if df_key not in self.handler._data:
            raise ValueError(f"Dataset '{df_key}' not found.")
        
        # ดึง DataFrame ที่ต้องการใช้งานออกมาจาก DataHandler
        df = self.handler.get_data(df_key)
        
        # สร้าง prefix สำหรับ prompt:
        # - รวมชื่อคอลัมน์ของ DataFrame
        # - รวมประเภทของข้อมูล (datatype) ของแต่ละคอลัมน์
        # - รวมคำแนะนำสำหรับการจัดรูปแบบ JSON จาก output_parser
        prefix = get_prefix(
            columns=', '.join(df.columns.tolist()),
            datatype=', '.join(f"{col}: {dtype}" for col, dtype in df.dtypes.to_dict().items()),
            json_format=self.output_parser.get_format_instructions()
        )

        # สร้าง suffix สำหรับ prompt (ส่วนท้ายของ prompt ที่อาจมีคำแนะนำเพิ่มเติม)
        suffix = get_suffix(
            columns=', '.join(df.columns.tolist()),
            datatype=', '.join(f"{col}: {dtype}" for col, dtype in df.dtypes.to_dict().items()),
        )

        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix, 
            suffix=suffix,
            verbose=True,
            allow_dangerous_code=True,  
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            chunk_size=None,
            max_rows=None,
            input_variables=["df"],
            handle_tool_error=True,
        )

    def extract_code_snippet(self, parsed_output: dict) -> str:
        """
        ฟังก์ชันสำหรับดึงโค้ด Python จากผลลัพธ์ที่ผ่านการ parse แล้ว
        Parameters:
            parsed_output (dict): ผลลัพธ์ที่ถูก parse จาก JSON output
        Returns:
            โค้ด Python ในรูปแบบ string หากมีอยู่ มิฉะนั้นคืนค่าสตริงว่าง
        """
        try:
            # พยายามดึงค่า 'code' จาก parsed_output หากไม่พบจะคืนค่าสตริงว่าง
            return parsed_output.get('code', '')
        except (AttributeError, KeyError):
            # หากเกิดข้อผิดพลาดในการเข้าถึงข้อมูล ให้บันทึก error และคืนค่าสตริงว่าง
            logging.error("No code found in the parsed output")
            return ''


    def run(self, query: str, dataset_key: str) -> dict:
        """
        ฟังก์ชันสำหรับประมวลผลคำสั่งของผู้ใช้:
          - สร้าง agent สำหรับชุดข้อมูลที่ระบุ
          - ส่ง query ไปให้ agent ประมวลผล
          - Parse และ validate ผลลัพธ์ที่ได้ให้อยู่ในรูปแบบ JSON ตามโมเดล PlotResponse
        Parameters:
            query (str): คำถามหรือคำสั่งที่ผู้ใช้ส่งเข้ามา
            dataset_key (str): คีย์ของชุดข้อมูลที่ต้องการใช้งาน
        Returns:
            dict ที่มี key "status" ระบุผลลัพธ์ (success/error) และ key "data" หรือ "message" สำหรับผลลัพธ์หรือข้อความ error
        """
        try:
            # สร้าง agent สำหรับชุดข้อมูลที่ระบุโดยใช้เมธอด create_agent
            agent = self.create_agent(dataset_key)
            df = self.handler.get_data(dataset_key)
            # แก้ไข: ลบเครื่องหมาย comma ท้ายบรรทัด เพื่อให้ columns และ datatype เป็น string แท้ๆ
            columns = ', '.join(df.columns.tolist())
            datatype = ', '.join(f"{col}: {dtype}" for col, dtype in df.dtypes.to_dict().items())
            
            # Construct the prompt template
            prompt_template = (f"""
                You are a Python expert specializing in data processing and analysis. 
                You are working with a DataFrame that has the following columns: {columns}, 
                and the corresponding data types: {datatype}. 
                Now, answer the following question: {query}
                IMPORTANT: If the user's query references a column or concept that is not present in the DataFrame,
                use your expertise to identify and substitute with the available column(s) that best match the intended analysis.
                available columns: {columns}"""
            )


            # Invoke the agent with the formatted prompt
            response = agent.invoke(prompt_template)
            
            try:
                # ตรวจสอบผลลัพธ์ที่ได้จาก agent:
                # หาก response['output'] เป็นสตริง ให้พยายามแปลงเป็น dict ด้วย json.loads
                if isinstance(response['output'], str):
                    parsed_output = json.loads(response['output'])
                else:
                    # หาก response['output'] อยู่ในรูปแบบ dict อยู่แล้ว ให้นำมาใช้งานโดยตรง
                    parsed_output = response['output']
                
                # Validate ผลลัพธ์ที่ได้โดยการสร้าง instance ของ PlotResponse
                validated_output = PlotResponse(**parsed_output)
                
                # คืนค่า output ที่ validate แล้วในรูปแบบ dict โดยใช้ model_dump() ของ Pydantic
                return {"status": "success", "data": validated_output.model_dump()}
            
            except json.JSONDecodeError as e:
                # หากเกิดข้อผิดพลาดในการแปลง JSON ให้บันทึก error และคืนค่า error message
                logging.error(f"Error parsing JSON: {e}")
                return {"status": "error", "message": f"JSON parsing error: {e}"}
            except Exception as e:
                # หากเกิดข้อผิดพลาดอื่น ๆ ในการประมวลผล output ให้บันทึก error และคืนค่า error message
                logging.error(f"Error processing output: {e}")
                return {"status": "error", "message": f"Processing error: {e}"}
            
        except Exception as e:
            # หากเกิดข้อผิดพลาดในขั้นตอนการสร้าง agent หรือส่ง query ให้บันทึก error และคืนค่า error message
            logging.error(f"An error occurred: {e}")
            return {"status": "error", "message": str(e)}


        
    def run_and_return_code(self, query: str, dataset_key: str) -> dict:
        """
        ฟังก์ชันสำหรับประมวลผล query และคืนค่าเฉพาะส่วนของโค้ดและคำอธิบาย
        Parameters:
            query (str): คำถามหรือคำสั่งจากผู้ใช้
            dataset_key (str): คีย์ของชุดข้อมูลที่ต้องการใช้งาน
        Returns:
            dict ที่ประกอบด้วย:
              - query: คำถามของผู้ใช้
              - code: โค้ด Python ที่ได้จากการประมวลผล (หลังจากตรวจสอบและแก้ไขความถูกต้องแล้ว)
              - explanation: คำอธิบายเกี่ยวกับการวิเคราะห์ที่ทำขึ้น
            หากเกิดข้อผิดพลาด จะคืนค่า dict ที่มี key "error" พร้อมข้อความ error
        """
        # # เรียกใช้เมธอด run() เพื่อประมวลผล query และรับผลลัพธ์ในรูปแบบ dict
        # result = self.run(query, dataset_key)
        # # หากสถานะของผลลัพธ์เป็น "success" ให้ดึงข้อมูล query, code และ explanation จากผลลัพธ์ที่ validate แล้ว
        # if result.get("status") == "success":
        #     data = result.get("data", {})

        #     return {
        #         "query": data.get('query'),
        #         "code": data.get("code"),
        #         "explanation": data.get("explanation")
        #     }
        # else:
        #     # หากสถานะไม่ใช่ success ให้คืนค่า dict ที่มี key "error" พร้อมข้อความ error ที่ได้รับ
        #     return {"error": result.get("message", "Unknown error")}
        try:
            result = self.run(query, dataset_key)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response"}
                    
            if isinstance(result, dict) and result.get("status") == "success":
                data = result.get("data", {})
                return {
                    "query": data.get('query'),
                    "code": data.get("code"),
                    "explanation": data.get("explanation")
                }
            return {"error": result.get("message", "Unknown error")}
        except Exception as e:
            return {"error": str(e)}
