import os
from typing import Optional                    
import pandas as pd            
import logging                 
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from datahandle_commented import DataHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json  
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from prompt import get_analysis_prompt
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
    explanation: str = Field(description="Detailed explanation Based on the provided data analysis")
    # Field สำหรับเก็บโค้ด Python ที่ agent สร้างขึ้นเพื่อให้ผู้ใช้สามารถนำไปใช้งานได้
    code: Optional[str] = Field(default=None, description="The analysis of the topics discussed")

# =======================================================================
# PandasAgent สำหรับจัดการการทำงานเกี่ยวกับ DataFrame และสร้าง agent
# =======================================================================
class AnalyseAgent:
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
        สร้าง agent สำหรับวิเคราะห์และตอบคำถามเชิงปริมาณจากข้อมูล
        """
        json_format=self.output_parser.get_format_instructions()
        df = self.handler.get_data(df_key)
        if df is None or df.empty:
            raise ValueError(f"ไม่พบข้อมูลสำหรับ key: {df_key}")

        prompt = get_analysis_prompt(df, json_format)

        # สร้าง agent พร้อมกับ opt-in ให้ execute dangerous code
        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            prompt=prompt,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="generate",
            return_intermediate_steps=True,
            chunk_size=None,
            max_rows=None,
            input_variables=["df"],
            handle_tool_error=True,
            allow_dangerous_code=True
            )

    def run(self, query: str, dataset_key: str) -> dict:
        try:
            agent = self.create_agent(dataset_key)
            df = self.handler.get_data(dataset_key)
            
            enhanced_query = f"""
            ANALYSIS REQUEST: {query}
            
            Please provide:
            ** Key insights **
            - Direct numerical answers
            - Relevant statistics
            - Clear comparisons if applicable
            - Brief explanation of findings
            
            Dataset Context:
            - Total Records: {len(df)}
            
            - VERY IMPORTANT: Do NOT use a sample of the dataset. You MUST analyze the FULL dataset.
                - Count and consider ALL rows.
                - Do NOT summarize or use only a small portion.
                - If dataset is too large, provide exact numbers, not estimates.

            
            Focus on providing quantitative insights directly from the data.
            """.strip()
            
            response = agent.invoke({"input": enhanced_query})
            
            output_text = response.get('output', '').strip()
            if not output_text:
                raise ValueError("Empty response from analysis agent")
            
            try:
                parsed_output = json.loads(output_text)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON: {e}")
                # ใช้ output_text โดยตรงเป็น string สำหรับ explanation
                parsed_output = {
                    "query": query,
                    "code": "",  # ไม่มีโค้ดในกรณีนี้
                    "explanation": output_text  # ใช้ output_text เป็น string
                }
            
            validated_output = PlotResponse(**parsed_output)
            return {
                "status": "success", 
                "data": {
                    "query": validated_output.query,
                    "code": validated_output.code,
                    "explanation": {
                        "text": validated_output.explanation
                    }
                }
            }
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return {"status": "error", "message": str(e)}




    def run_and_return_code(self, query: str, dataset_key: str) -> dict:
        result = self.run(query, dataset_key)
        if result.get("status") == "success":
            data = result.get("data", {})
            return {
                "query": query,
                "response": json.dumps({
                    "code": data.get("code"),
                    "explanation": data.get("explanation")
                })
            }
        else:
            return {"error": result.get("message", "Unknown error")}

