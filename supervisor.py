# -----------------------------------------------------------------------
# main script สำหรับ Suopervisor Agent ที่ใช้ในการสร้าง agent และติดต่อกับโมเดลภาษา
# -----------------------------------------------------------------------
import locale
import traceback
                 
try:
    locale.setlocale(locale.LC_ALL, 'th_TH.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

import logging
import re                       
import sys                      
from typing import Optional     

# กำหนดให้การแสดงผลบน stdout รองรับการเข้ารหัสแบบ utf-8
# sys.stdout.reconfigure(encoding='utf-8')

import contextlib               
import io                       
import json                     
from dotenv import load_dotenv  
import os                       
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from pandas_agent import PandasAgent
from datahandle import DataHandler
from analys_agent import AnalyseAgent 
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from datetime import datetime
import pytz                     # สำหรับจัดการ timezone
from langchain_core.prompts import PromptTemplate
import seaborn as sns
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from prompt import get_react_prompt, get_explanation_prompt, get_run_prompt

# โหลด environment variables จากไฟล์ .env
load_dotenv()

# กำหนดค่าคงที่สำหรับเส้นทางของไฟล์ static และกราฟที่สร้างขึ้น
STATIC_DIR = "static"  # โฟลเดอร์สำหรับไฟล์ static
PLOT_DIR = os.path.join(STATIC_DIR, "plots")  # โฟลเดอร์สำหรับเก็บไฟล์กราฟ
PLOT_URL_PREFIX = "/static/plots"  # URL prefix สำหรับเข้าถึงกราฟผ่านส่วน frontend

# -----------------------------------------------------------------------
# ส่วนของโมเดลสำหรับเก็บข้อมูลต่างๆ ที่จะใช้ในการส่งและรับผลลัพธ์จาก agent
# -----------------------------------------------------------------------

class PlotInfo(BaseModel):
    """
    โมเดลสำหรับเก็บข้อมูลของไฟล์กราฟที่ถูกสร้างขึ้น
    Attributes:
        filename (str): ชื่อไฟล์กราฟ
        path (str): เส้นทางของไฟล์กราฟสำหรับเข้าถึงผ่านเว็บ
        created_at (str): เวลาที่สร้างไฟล์กราฟ (ในรูปแบบ string)
    """
    filename: str
    path: str
    created_at: str

class ExecutionResult(BaseModel):
    """
    โมเดลสำหรับเก็บผลลัพธ์จากการรันโค้ด Python
    Attributes:
        output (Optional[str]): ข้อความผลลัพธ์ที่ได้จากการรันโค้ด (ถ้ามี)
        error (Optional[str]): ข้อความ error ที่เกิดขึ้นระหว่างการรันโค้ด (ถ้ามี)
        plots (List[PlotInfo]): รายการของกราฟที่ถูกสร้างขึ้นระหว่างการรันโค้ด
    """
    output: Optional[str] = None
    error: Optional[str] = None
    plots: List[PlotInfo] = []

class SubResponseContent(BaseModel):
    """
    โมเดลสำหรับเก็บข้อมูลย่อยของผลลัพธ์ที่ได้จากการเรียกใช้งาน tool
    Attributes:
        code (Optional[str]): โค้ดที่ถูกสร้างขึ้นหรือใช้ในการวิเคราะห์ (ถ้ามี)
        execution_result (Optional[ExecutionResult]): ผลลัพธ์จากการรันโค้ด (ถ้ามี)
        explanation (Optional[Dict[str, Any]]): คำอธิบายหรือผลลัพธ์เพิ่มเติมจากการวิเคราะห์ (ถ้ามี)
        type (str): ประเภทของผลลัพธ์ (ค่าเริ่มต้น "tool_response")
        response (Optional[str]): ข้อความตอบกลับที่ได้จาก tool (ถ้ามี)
    """
    code: Optional[str] = None
    execution_result: Optional[ExecutionResult] = None
    explanation: Optional[Dict[str, Any]] = None
    type: str = "tool_response"
    response: Optional[str] = None

class MetaData(BaseModel):
    """
    โมเดลสำหรับเก็บข้อมูล metadata ของการตอบกลับ
    Attributes:
        timestamp (str): เวลาที่ตอบกลับ (ในรูปแบบ string)
        model (str): ชื่อโมเดลที่ใช้ในการประมวลผล
        temperature (float): ค่า temperature ที่ใช้ในโมเดล
        tools_used (List[str]): รายชื่อเครื่องมือ (tools) ที่ถูกเรียกใช้งาน
        dataset_key (str): คีย์ของชุดข้อมูลที่ใช้งาน
        status (str): สถานะของการประมวลผล (ค่าเริ่มต้น "success")
    """
    timestamp: str
    model: str
    temperature: float
    tools_used: List[str]
    dataset_key: str
    status: str = "success"

class SupervisorResponse(BaseModel):
    """
    โมเดลหลักสำหรับการตอบกลับจาก SupervisorAgent
    Attributes:
        query (str): คำถามหรือคำสั่งที่ได้รับจากผู้ใช้
        response (str): คำตอบหรือผลลัพธ์หลักที่ได้จาก agent
        raw_response (Optional[str]): ข้อมูล raw response จาก agent (ถ้ามี)
        sub_response (Dict[str, SubResponseContent]): ผลลัพธ์ย่อยที่ได้จากการเรียกใช้งานเครื่องมือต่างๆ
        plot_data (Dict[str, List[PlotInfo]]): ข้อมูลของกราฟที่ถูกสร้างขึ้น
        metadata (MetaData): ข้อมูล metadata ของการตอบกลับ
        error (Optional[str]): ข้อความ error หากมีข้อผิดพลาดเกิดขึ้น
    """
    query: str
    response: str
    raw_response: Optional[str] = None  # ข้อมูล raw response ที่ได้จาก agent
    sub_response: Dict[str, SubResponseContent]
    plot_data: Dict[str, List[PlotInfo]]
    metadata: MetaData
    error: Optional[str] = None

    class Config:
        # อนุญาตให้ใช้ชนิดข้อมูลแบบ arbitrary types
        arbitrary_types_allowed = True
        # กำหนดตัวเข้ารหัสสำหรับ datetime ให้แปลงเป็น ISO format
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PlotResponse(BaseModel):
    """
    โมเดลสำหรับการตอบกลับในส่วนที่เกี่ยวกับการวิเคราะห์และแสดงผลกราฟ
    Attributes:
        query (str): คำอธิบายเกี่ยวกับคำถามของผู้ใช้
        response (str): คำตอบหรือผลลัพธ์ที่ได้จาก SupervisorAgent
        sub_response (dict): ผลลัพธ์จากเครื่องมือ (tool) ที่ SupervisorAgent ใช้งาน
    """
    query: str = Field(description="Description of what is user query")
    response: str = Field(description="Description of output from SupervisorAgent")
    sub_response: dict = Field(description="Response from tool that SupervisorAgent uses")

    def dict(self, *args, **kwargs):
        """
        ปรับปรุงเมธอด dict() เพื่อ customize รูปแบบของ output
        """
        return {
            "query": self.query,
            "response": self.response,
            "sub_response": self.sub_response
        }

# -----------------------------------------------------------------------
# คลาสหลักสำหรับ SupervisorAgent
# คลาสนี้จะรวบรวมฟังก์ชันสำหรับการติดต่อกับโมเดลภาษา,
# จัดการ memory, เรียกใช้งานเครื่องมือวิเคราะห์ข้อมูล (PandasAgent),
# รันโค้ด, และสร้างผลลัพธ์ตอบกลับในรูปแบบที่มีโครงสร้าง
# -----------------------------------------------------------------------

class SupervisorAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, dataset_paths: dict, dataset_key: str, session_id: str, 
                 supervisor_api_key: str, agent_api_key: str, explanner_api_key: str):
        """
        ตัวสร้าง (constructor) สำหรับ SupervisorAgent
        Parameters:
            temperature (float): ค่า temperature ที่ใช้ในการประมวลผลของโมเดล
            base_url (str): URL พื้นฐานสำหรับเรียกใช้งาน API
            model_name (str): ชื่อของโมเดลภาษา (LLM)
            dataset_paths (dict): ข้อมูลหรือเส้นทางของชุดข้อมูลที่จะใช้งาน
            dataset_key (str): คีย์ที่ระบุชุดข้อมูลที่ใช้งาน
            session_id (str): รหัส session สำหรับติดตามการสนทนา
            supervisor_api_key (str): API key สำหรับ supervisor (LLM หลัก)
            agent_api_key (str): API key สำหรับ PandasAgent
            explanner_api_key (str): API key สำหรับ LLM ย่อยที่ใช้ให้คำอธิบาย
        """
        # กำหนดค่า parameter ที่ได้รับให้กับ attribute ของ instance
        self.temperature = temperature
        self.base_url = base_url
        self.model = model_name
        self.dataset_key = dataset_key

        # เก็บ API keys สำหรับการเรียกใช้งานโมเดลและเครื่องมือต่างๆ
        self.api_key = supervisor_api_key
        self.api_sub_key = explanner_api_key
        self.pandas_api = agent_api_key

        self.session_id = session_id
        # เริ่มต้นโมเดลภาษา (LLM) หลัก
        self.llm = self.initialize_llm()
        # เริ่มต้นโมเดลภาษา (LLM) ย่อยสำหรับให้คำอธิบาย
        self.llms = self.initialize_sub_llm()
        # ตั้งค่า memory สำหรับเก็บประวัติการสนทนา
        self.memory = self.initialize_memory()
        # สร้าง instance ของ PandasAgent สำหรับจัดการและวิเคราะห์ข้อมูล DataFrame
        self.pandas_agent = PandasAgent(temperature, base_url, model_name, dataset_paths, session_id, api_pandas_key=self.pandas_api)
        self.analysis_agent = AnalyseAgent(temperature, base_url, model_name, dataset_paths, session_id, api_pandas_key=self.pandas_api)

        # กำหนดเครื่องมือ (tools) ที่จะใช้ในการประมวลผล (spandas_agent)
        self.tools = self.initialize_tools()
        # สร้าง agent โดยใช้ prompt และเครื่องมือที่เตรียมไว้
        self.agent = self.create_agent()
        # สร้าง executor สำหรับจัดการ query และการประมวลผลของ agent
        self.agent_executor = self.create_agent_executor()
        # กำหนดตัว parser สำหรับแปลงผลลัพธ์ให้อยู่ในรูปแบบ JSON
        self.output_parser = JsonOutputParser()

    def initialize_llm(self) -> ChatOpenAI:
        """
        ฟังก์ชันสำหรับเริ่มต้นและตั้งค่าโมเดลภาษา (LLM) หลัก
        Returns:
            instance ของ ChatOpenAI ที่ถูกกำหนดค่าแล้ว
        """
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            top_p=0.95  # กำหนด top_p สำหรับการควบคุมการสุ่มเลือก token
        )
    
    def initialize_sub_llm(self) -> ChatOpenAI:
        """
        ฟังก์ชันสำหรับเริ่มต้นและตั้งค่าโมเดลภาษา (LLM) ย่อย
        ซึ่งใช้สำหรับให้คำอธิบายผลลัพธ์เพิ่มเติม
        Returns:
            instance ของ ChatOpenAI ที่ถูกกำหนดค่าแล้วสำหรับงานย่อย
        """
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_sub_key,
            temperature=self.temperature,
        )

    def initialize_memory(self):
        """
        ฟังก์ชันสำหรับตั้งค่า memory buffer เพื่อเก็บประวัติการสนทนา
        Returns:
            instance ของ ConversationBufferMemory ที่เก็บ chat_history
        """
        return ConversationBufferMemory(memory_key="chat_history", return_messages=False)
    
    def clear_memory(self):
        """
        ฟังก์ชันสำหรับล้าง memory ของการสนทนา
        ใช้เมื่อต้องการเริ่มการสนทนาใหม่
        """
        print('Memory cleared !!!')
        return self.memory.clear()

#================================================================================================
    def initialize_tools(self):
        """
        ฟังก์ชันสำหรับตั้งค่าและกำหนดเครื่องมือ (tools) ที่ TyphoonAgent จะใช้งาน
        ในที่นี้มีเครื่องมือสำหรับการวิเคราะห์ข้อมูลผ่าน PandasAgent
        Returns:
            รายการของเครื่องมือ (list) ที่ใช้งาน
        """

        analysis_tool = Tool(
            name="analysis_agent",
            func=self.query_analysis,
            description=(
                "This tool is best suited for in-depth statistical analysis, hypothesis testing, correlation studies, "
                "and extracting general insights from the dataset. "
                "Use this tool if the query involves identifying relationships, computing statistical metrics, "
                "detecting trends, performing predictive modeling, or any task requiring a deeper interpretation of the data. "
                "**Do not use this tool if the query requires a direct DataFrame operation, visualization, or structured table output.** "
                "If the question does not explicitly mention a need for charts or tables, prioritize using this tool."
            ).strip(),
        )
        pandas_tool = Tool(
            name="pandas_agent",
            func=self.query_dataframe,
            description=(
                "Use this tool **only** if the user explicitly requests: "
                "- A chart, graph, or plot (e.g., line chart, bar chart, histogram, etc.). "
                "- A structured table or formatted DataFrame output. "
                "- Direct DataFrame operations such as filtering, grouping, sorting, or aggregation. "
                "**Do not use this tool for high-level statistical insights or explanatory analysis.** "
                "If the task requires calculations beyond simple aggregations (e.g., hypothesis testing, correlation analysis), "
                "redirect the query to the `analysis_agent` instead."
            ).strip(),
        )

        return [analysis_tool, pandas_tool]
        # return [analysis_tool]
#================================================================================================
    
    def query_dataframe(self, user_input: str) -> dict:
        """
        ฟังก์ชันสำหรับส่งคำสั่งที่เกี่ยวกับการวิเคราะห์ข้อมูลไปยัง PandasAgent
        Parameters:
            user_input (str): คำสั่งหรือคำถามเกี่ยวกับข้อมูลที่ผู้ใช้ต้องการ
        Returns:
            dict ที่มีผลลัพธ์จากการประมวลผล (เช่น โค้ด Python ที่สร้างขึ้น หรือ error message)
        """
        try:
            # เรียกใช้ฟังก์ชัน run_and_return_code ของ PandasAgent พร้อมส่งคำสั่งและ dataset key
            result = self.pandas_agent.run_and_return_code(user_input, self.dataset_key)
            if 'code' in result:
                # กำจัดคำสั่ง plt.show() ออกเพราะจะทำให้เกิดปัญหาเมื่อรันในสภาพแวดล้อม backend
                result['code'] = result['code'].replace('plt.show()', '')
                
                # หากโค้ดมีการสร้าง figure และมีการ plot กราฟ
                if 'plt.figure' in result['code'] and '.plot(' in result['code']:
                    # กำจัดการตั้งค่า figsize ที่อาจซ้ำซ้อน
                    result['code'] = result['code'].replace('plt.figure(figsize=(10, 6))\n', '')
                    # เพิ่มการตั้งค่า figsize ให้กับ plot ในกรณีที่ไม่มีการระบุไว้
                    result['code'] = result['code'].replace('.plot(', '.plot(figsize=(10, 6), ')
                    
            return result
        except Exception as e:
            # กรณีเกิดข้อผิดพลาดจะส่งกลับ dict ที่มีข้อความ error
            return {
                "error": str(e),
                "query": user_input,
                "explanation": "Error occurred while processing the query"
            }
        
    def query_analysis(self, user_input: str) -> dict:
        """
        Function to send analysis-related commands to the analysis_agent.
        
        Parameters:
            user_input (str): The command or question related to the data that the user wants to analyze.
        
        Returns:
            dict: A dictionary containing the results of the analysis. The dictionary includes:
                - "query": The original user input.
                - "explanation": Detailed explanation of the analysis.
                - "error" (optional): Error message if an error occurred during processing.
        """
        try:
            # Call the run_and_return_code method of the analysis_agent with the user input and dataset key
            result = self.analysis_agent.run_and_return_code(user_input, self.dataset_key)
            
            # Check if the result contains an error
            if 'error' in result:
                logging.error(f"Error in analysis: {result['error']}")
                return {
                    "error": result['error'],
                    "query": user_input,
                    "explanation": "Error occurred while processing the query"
                }
            
            # Return the successful result
            return result
        
        except Exception as e:
            # Log the exception and return an error message
            logging.error(f"Exception occurred during analysis: {str(e)}")
            return {
                "error": str(e),
                "query": user_input,
                "explanation": "Exception occurred while processing the query"
            }

#================================================================================================

    def create_agent(self):
        """
        ฟังก์ชันสำหรับสร้าง agent แบบ React agent โดยใช้ข้อมูลจาก dataset
        และ prompt ที่มีข้อมูลของ dataset
        Returns:
            instance ของ agent ที่ถูกสร้างขึ้น
        """
        # ตรวจสอบว่าชุดข้อมูลที่ระบุ (dataset_key) มีอยู่ใน PandasAgent หรือไม่
        if self.dataset_key not in self.pandas_agent.handler._data:
            raise ValueError(f"Dataset '{self.dataset_key}' not found.")

        # ดึง DataFrame ของชุดข้อมูลออกมา
        df = self.pandas_agent.handler.get_data(self.dataset_key)
        
        # สร้าง prompt สำหรับ agent โดยส่งข้อมูลคีย์และคอลัมน์ของ DataFrame
        react_prompt = get_react_prompt(dataset_key=self.dataset_key, 
                                        df_columns=df.columns)
        
        # สร้าง agent โดยใช้โมเดลภาษาหลัก (LLM) เครื่องมือที่กำหนด และ prompt ที่สร้างขึ้น
        return create_react_agent(llm=self.llm, 
                                  tools=self.tools, 
                                  prompt=react_prompt,
                                  )

    def create_agent_executor(self):
        """
        ฟังก์ชันสำหรับสร้าง executor ที่จะจัดการ query และการดำเนินการของ agent
        Returns:
            instance ของ AgentExecutor ที่ถูกกำหนดค่าไว้
        """
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,  # เปิดการแสดง log รายละเอียดสำหรับการดีบัก
            max_iterations=int(os.getenv("MAX_ITERATIONS", 20)),  # จำนวน iteration สูงสุดสำหรับการประมวลผล
            handle_parsing_errors=True,  # จัดการ error ในการ parse output จาก agent
            return_intermediate_steps=True ,
        )

    def _prepare_plot_code(self, code: str) -> str:
        """
        ฟังก์ชันภายในสำหรับเตรียมโค้ด Python ที่เกี่ยวกับการ plot
        โดยจะลบคำสั่ง plt.show() ออกเพื่อป้องกันการแสดงผลใน backend
        Parameters:
            code (str): โค้ด Python ที่ต้องการประมวลผล
        Returns:
            โค้ดที่ปรับปรุงแล้ว
        """
        code = code.replace('plt.show()', '')
        return code
    

    # try this code bellow 
    def execute_code(self, code: str) -> ExecutionResult:
        # ปิดกราฟที่อาจเปิดอยู่ก่อนหน้าเพื่อให้เริ่มต้นใหม่
        plt.close('all')
        
        # สร้างโฟลเดอร์สำหรับเก็บไฟล์กราฟ หากยังไม่มีอยู่
        plot_dir = "static/plots"
        os.makedirs(plot_dir, exist_ok=True)
        # สร้างตัวแปร current_time สำหรับตั้งชื่อไฟล์กราฟที่ไม่ซ้ำกัน
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # สร้าง context สำหรับรันโค้ด ซึ่งประกอบด้วยโมดูลและ DataFrame ที่จำเป็น
        context = {
            "pd": pd, 
            "np": np, 
            "sns": sns, 
            "plt": plt, 
            "tabulate": tabulate, 
            "df": self.pandas_agent.handler.get_data(self.dataset_key)
        }
        
        # สร้าง StringIO object สำหรับจับ output จากการรันโค้ด
        output = io.StringIO()
        # รายการสำหรับเก็บข้อมูลของกราฟที่สร้างขึ้น
        plot_files = []
        
        # Redirect stdout ไปยัง output เพื่อจับข้อความที่พิมพ์ออกมาในระหว่างการรันโค้ด
        with contextlib.redirect_stdout(output):
            try:
                # รันโค้ดที่ได้รับมาใน context ที่กำหนด
                exec(code, context)
                
                # ตรวจสอบกราฟทั้งหมดที่ถูกสร้างขึ้น
                fig_nums = plt.get_fignums()
                for i, fig_num in enumerate(fig_nums):
                    fig = plt.figure(fig_num)
                    # สร้างชื่อไฟล์กราฟที่มีหมายเลข index เพื่อป้องกันชื่อซ้ำ
                    filename = f"plot_{current_time}_{i+1}.png"
                    filepath = os.path.join(plot_dir, filename)
                    
                    # บันทึกกราฟลงในไฟล์ด้วยคุณภาพสูง
                    fig.savefig(filepath, bbox_inches='tight', dpi=300, format='png')
                    
                    # สร้างข้อมูลของกราฟในรูปแบบ PlotInfo
                    plot_files.append(PlotInfo(
                        filename=filename,
                        path=f"/static/plots/{filename}",
                        created_at=datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')
                    ))
                    # ปิดกราฟเพื่อปล่อยหน่วยความจำ
                    plt.close(fig)
                    
                # ส่งกลับผลลัพธ์การรันโค้ดในรูปแบบ ExecutionResult
                return ExecutionResult(
                    output=output.getvalue(),
                    plots=plot_files
                )
                
            except Exception as e:
                # หากเกิดข้อผิดพลาด ให้ปิดกราฟทั้งหมดและส่งกลับ error message
                plt.close('all')
                return ExecutionResult(
                    error=str(e),
                    plots=[]
                )


    def get_explanation(self, output, user_input) -> dict:
        """
        ฟังก์ชันสำหรับขอคำอธิบายของ output จากโมเดลภาษา (LLM) ย่อย
        Parameters:
            output: ข้อความ output ที่ต้องการให้ LLM ช่วยอธิบาย
            user_input: คำถามดั้งเดิมของผู้ใช้
        Returns:
            dict ที่ประกอบด้วยคำอธิบายหรือ error message หากเกิดปัญหา
        """
        # สร้าง prompt สำหรับการขอคำอธิบาย โดยส่งตัว output_parser ไปด้วย
        prompt = get_explanation_prompt(output_parser=self.output_parser)
        try:
            # สร้าง chain การประมวลผลโดยใช้ prompt, LLM ย่อย และ output parser
            chain = prompt | self.llms | self.output_parser
            response = chain.invoke({
                "output": output,
                "user_question": user_input
            })
            explanation = response
        except Exception as e:
            # กรณีเกิดข้อผิดพลาดให้ส่งกลับ error message พร้อมกับ raw output
            explanation = {"error": f"Error getting explanation: {e}", "raw_output": output}
        
        # ตรวจสอบให้แน่ใจว่าคำอธิบายอยู่ในรูปแบบ JSON ที่ถูกต้อง
        if isinstance(explanation, str):
            try:
                json_explanation = json.loads(explanation)
            except json.JSONDecodeError:
                json_explanation = {"error": "Invalid JSON output", "raw_output": explanation}
        elif isinstance(explanation, dict):
            json_explanation = explanation
        else:
            json_explanation = {"error": "Unexpected output type", "raw_output": explanation}
        
        return json_explanation

    def _process_tool_output(self, output) -> Dict[str, Any]:
        """
        ฟังก์ชันสำหรับประมวลผล output จากเครื่องมือ (tool)
        ให้เป็นรูปแบบที่เป็นมาตรฐาน (standardized format)
        Parameters:
            output: ผลลัพธ์จากเครื่องมือที่อาจเป็น string หรือ dict
        Returns:
            dict ที่มี key "response" และ "type" เพื่อระบุประเภทของ output
        """
        # if isinstance(output, dict):
        #     return output
        # elif isinstance(output, str):
        #     # หาก output ดูเหมือนเป็น JSON ให้พยายามแปลงเป็น dict
        #     if output.strip().startswith('{'):
        #         try:
        #             return json.loads(output)
        #         except json.JSONDecodeError:
        #             return {"response": output, "type": "text_response"}
        #     else:
        #         return {"response": output, "type": "text_response"}
        # else:
        #     return {"response": str(output), "type": "converted_response"}
        if isinstance(output, str):
            # ลบ whitespace และ newline ที่ไม่จำเป็น
            output = output.strip()
            try:
                # พยายาม parse JSON
                return json.loads(output)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {e}")
                # ถ้า parse ไม่ได้ ให้ส่งกลับในรูปแบบ dict ที่มี response field
                return {"response": output}
        return output
        


    def run(self, user_input: str) -> SupervisorResponse:
        """
        ฟังก์ชันหลักสำหรับการรัน agent และประมวลผลคำสั่งของผู้ใช้
        โดยจะประสานงานระหว่างการเรียกใช้งานโมเดลภาษา, การวิเคราะห์ข้อมูล,
        การรันโค้ด, และการรวมผลลัพธ์เข้าด้วยกันในรูปแบบที่มีโครงสร้าง
        Parameters:
            user_input (str): คำสั่งหรือคำถามจากผู้ใช้
        Returns:
            instance ของ SupervisorResponse ที่ประกอบด้วยผลลัพธ์, metadata,
            ข้อมูลของกราฟ (ถ้ามี) และรายละเอียดของขั้นตอนการประมวลผล
        """
        try:
            logging.info(f"Running SupervisorAgent with input: {user_input}")
            
            # ดึงข้อมูล DataFrame จาก PandasAgent ตาม dataset key ที่ระบุ
            df = self.pandas_agent.handler.get_data(self.dataset_key)
            # สร้าง prompt สำหรับรันคำสั่ง โดยรวมคำสั่งของผู้ใช้เข้ากับข้อมูลของ DataFrame
            input_query = get_run_prompt(dataset_key=self.dataset_key, 
                                         df_columns=df.columns).format(user_input=user_input)

            # ดึง raw response จาก agent โดยเก็บ verbose output เพื่อติดตามขั้นตอนภายใน
            verbose_output = io.StringIO()
            with contextlib.redirect_stdout(verbose_output):
                raw_response = self.agent_executor.invoke({"input": user_input}, verbose=True)
            
            # ดึงผลลัพธ์หลักจาก raw response
            main_response = raw_response.get('output', '')
            # เก็บ intermediate steps ที่เกิดขึ้นระหว่างการประมวลผล
            intermediate_steps = raw_response.get('intermediate_steps', [])
            # เตรียม dict สำหรับเก็บผลลัพธ์ย่อยจากเครื่องมือแต่ละตัว
            sub_response = {}
            # เตรียม dict สำหรับเก็บข้อมูลของกราฟ
            plot_data = {"plots": []}
            # เก็บ verbose content ทั้งหมดที่ได้จากการรัน agent
            verbose_content = verbose_output.getvalue()
            verbose_output.close()

            # ใช้ regular expression ในการดึงข้อมูลที่เป็น "Thought:" จาก verbose content
            pattern = r"Thought:.*?Finished chain\..*"

            # ค้นหาข้อความที่ตรงกับ pattern
            match = re.search(pattern, verbose_content, re.DOTALL)
            if match:
                extracted_text = match.group(0)
                # ตัดส่วนที่เกินออกหลังคำว่า "Finished chain."
                extracted_text = extracted_text.rsplit("Finished chain.", 1)[0].strip()
                # ลบ escape sequences ที่อาจเกิดจากการแสดงผลในเทอร์มินัลออก
                extracted_text = re.sub(r"\[\d+[;]?\d*m|> ?", "", extracted_text)
            else:
                extracted_text = verbose_content

            # วนลูปผ่าน intermediate steps ที่ได้จาก agent
            for step in intermediate_steps:
                if len(step) >= 2:
                    # ดึงชื่อเครื่องมือที่ถูกเรียกใช้งาน
                    tool_name = step[0].tool
                    # ประมวลผล output จากเครื่องมือให้เป็น dict ที่มีรูปแบบมาตรฐาน
                    tool_output = self._process_tool_output(step[1])
                    
                    # หากเครื่องมือที่เรียกใช้คือ pandas_agent และมีการส่งโค้ดกลับมา
                    if tool_name == "pandas_agent" and "code" in tool_output:
                        # รันโค้ดที่ได้จาก tool
                        # execution_result = self.execute_code(tool_output["code"])
                        code_snippet = tool_output.get("code", "")
                        if not code_snippet:
                            return ExecutionResult(error="No code found in tool output", plots=[])
                        execution_result = self.execute_code(code_snippet)
                        # ขอคำอธิบายของ output หรือ error จากการรันโค้ด
                        explanation = self.get_explanation(
                            execution_result.output if execution_result.output 
                            else execution_result.error,
                            user_input
                        )
                        
                        # เก็บผลลัพธ์จากเครื่องมือ pandas_agent ในรูปแบบของ SubResponseContent
                        sub_response[tool_name] = SubResponseContent(
                            code=tool_output["code"],
                            execution_result=execution_result,
                            explanation=explanation
                        )
                        
                        # หากมีกราฟที่ถูกสร้างขึ้น ให้นำข้อมูลของกราฟมาเก็บใน plot_data
                        if execution_result.plots:
                            plot_data["plots"].extend(execution_result.plots)


                    elif tool_name == "analysis_agent":
                        try:
                            analysis_result = json.loads(tool_output.get("response", ""))
                        except json.JSONDecodeError as e:
                            logging.error(f"Error parsing JSON: {e}")
                            analysis_result = {}
                        code_val = analysis_result.get("code", "")
                        explanation_val = analysis_result.get("explanation", "")
                        # หาก explanation ไม่ใช่ dict ให้ห่อหุ้มเป็น dict ด้วย key "text"
                        if not isinstance(explanation_val, dict):
                            explanation_val = {"text": explanation_val} if explanation_val else {}
                        sub_response[tool_name] = SubResponseContent(
                            code=code_val,
                            explanation=explanation_val,
                            type="tool_response"
                        )


            # สร้าง metadata สำหรับการตอบกลับ
            metadata = MetaData(
                timestamp=datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                model=self.model,
                temperature=self.temperature,
                tools_used=list(sub_response.keys()),
                dataset_key=self.dataset_key
            )

            # สร้างและส่งกลับผลลัพธ์ในรูปแบบ SupervisorResponse
            response = SupervisorResponse(
                query=input_query,
                response=main_response,
                raw_response=extracted_text,
                sub_response=sub_response,
                plot_data=plot_data,
                metadata=metadata
            )
            
            logging.info(f"SupervisorAgent response: {response.model_dump()}")
            return response

        except Exception as e:
            # หากเกิดข้อผิดพลาดในการประมวลผล ให้สร้าง metadata ที่มีสถานะ error
            error_metadata = MetaData(
                timestamp=datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                model=self.model,
                temperature=self.temperature,
                tools_used=[],
                dataset_key=self.dataset_key,
                status="error"
            )
            
            
            # ส่งกลับผลลัพธ์ที่มี error message
            logging.error(f"Error in SupervisorAgent run method: {str(e)}")
            logging.error(traceback.format_exc())
            return SupervisorResponse(
                query=input_query,
                response="Error occurred during processing",
                sub_response={},
                plot_data={"plots": []},
                metadata=error_metadata,
                error=str(e)
            )
