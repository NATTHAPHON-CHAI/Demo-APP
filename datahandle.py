import os
import re  
import pandas as pd  
import logging  
from dateutil.parser import parse  # สำหรับ fallback ในการแปลงวันที่

# ตั้งค่า logging ให้แสดง log ระดับ INFO และกำหนดรูปแบบข้อความ log ให้แสดงวันที่ เวลา ระดับ log และข้อความ
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataHandler:
    """
    คลาส DataHandler สำหรับจัดการการโหลดและ preprocess ข้อมูลจากไฟล์
    โดยใช้แนวคิด Singleton pattern เพื่อให้มี instance เดียวในระบบ
    """
    _instance = None  # ตัวแปรคลาสเพื่อเก็บ instance เดียวของ DataHandler

    def __new__(cls, *args, **kwargs):
        """
        ฟังก์ชัน __new__ จะถูกเรียกใช้ก่อน __init__ เพื่อสร้าง instance
        ที่นี่ใช้เพื่อบังคับให้มีแค่ instance เดียว (singleton)
        """
        if not cls._instance:
            # หากยังไม่มี instance ให้สร้าง instance ใหม่ด้วย super().__new__()
            cls._instance = super(DataHandler, cls).__new__(cls)
        return cls._instance  # คืนค่า instance เดียวให้กับทุกการเรียกใช้

    def __init__(self, dataset_paths=None):
        """
        ตัวสร้างสำหรับ DataHandler
        Parameters:
            dataset_paths: พจนานุกรมที่มี key เป็น identifier (เช่น "df1", "df2")
                           และ value เป็นเส้นทางไฟล์ของ dataset นั้น ๆ
        """
        # ตรวจสอบว่า instance นี้ถูก initial แล้วหรือยัง เพื่อป้องกันการรัน __init__ ซ้ำ
        if not hasattr(self, '_initialized'):
            self._initialized = True  # กำหนด flag ว่า instance ได้รับการ initial แล้ว
            if dataset_paths is None:
                dataset_paths = {}  # หากไม่มีการส่ง dataset_paths เข้ามา ให้ใช้ dict ว่าง
            self.dataset_paths = dataset_paths  # เก็บพจนานุกรมของ dataset paths ไว้ใน attribute ของ instance
            self._data = {}  # สร้าง attribute สำหรับเก็บข้อมูล DataFrame ที่โหลดมา

    def load_data(self) -> None:
        """
        ฟังก์ชันสำหรับโหลดข้อมูลจากทุกเส้นทางใน dataset_paths
        พร้อมทั้งทำการ standardize ชื่อคอลัมน์ใน DataFrame ให้เป็น lowercase และใช้ _ แทนช่องว่าง
        """
        # หากไม่มี dataset paths ให้โยนข้อผิดพลาด
        if not self.dataset_paths:
            raise ValueError("No dataset paths provided.")

        # วนลูปผ่านพจนานุกรม dataset_paths โดย key เป็น identifier และ dataset_path เป็นเส้นทางไฟล์
        for key, dataset_path in self.dataset_paths.items():
            # ตรวจสอบว่าไฟล์ที่ระบุมีอยู่จริงหรือไม่
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")

            # แยกส่วนชื่อไฟล์และนามสกุลออกจาก dataset_path
            _, ext = os.path.splitext(dataset_path)
            if ext == ".csv":
                try:
                    # หากเป็นไฟล์ CSV ให้ใช้ pd.read_csv อ่านไฟล์ด้วยการเข้ารหัส UTF-8
                    self._data[key] = pd.read_csv(dataset_path, encoding="utf-8")
                except UnicodeDecodeError:
                    # หากเกิดปัญหาในการ decode ด้วย UTF-8 ให้ลองใช้ encoding "latin1"
                    logging.warning(f"UTF-8 decoding failed for {key}. Trying 'latin1'.")
                    self._data[key] = pd.read_csv(dataset_path, encoding="latin1")
            elif ext == ".xls":
                self._data[key] = pd.read_excel(dataset_path, engine='xlrd')
            elif ext == ".xlsx":
                self._data[key] = pd.read_excel(dataset_path, engine='openpyxl')
            else:
                # หากนามสกุลไม่รองรับ ให้โยนข้อผิดพลาด
                raise ValueError(f"Unsupported file extension for {key}: {ext}")

            # ทำการ standardize ชื่อคอลัมน์ให้เป็น lowercase, ลบช่องว่างด้านหน้าและด้านหลัง และแทนที่ช่องว่างด้วย "_"
            self._data[key].columns = (
                self._data[key].columns.str.lower().str.strip().str.replace(" ", "_")
            )
            # บันทึก log แจ้งว่า dataset สำหรับ key นี้ถูกโหลดเรียบร้อยแล้ว พร้อมแสดงชื่อคอลัมน์
            logging.info(f"Data for {key} loaded. Columns: {', '.join(self._data[key].columns)}")



    def preprocess_data(self, threshold: float = 0.8, date_format: str = "%Y-%m-%d") -> None:
        """
        ฟังก์ชัน preprocess ข้อมูลใน dataset โดย:
        1. แปลงคอลัมน์วันที่ที่เป็น string เป็น datetime หาก non-NaT >= threshold
        2. แปลงคอลัมน์ string ที่มีตัวเลขเป็น numeric หาก non-NaN >= threshold

        Parameters:
            threshold (float): อัตราส่วนขั้นต่ำ (ค่าเริ่มต้น 0.8) ของค่าที่แปลงได้เพื่อให้การแปลงสำเร็จ
        """
        if not self._data:
            raise ValueError("Data not loaded.")

        id_pattern = r"id"  # regex สำหรับคอลัมน์ที่มี "id"

        for key, df in self._data.items():
            for col in df.columns:
                # ข้ามคอลัมน์ที่มี "id" ในชื่อ (ไม่สนใจ case)
                if re.search(id_pattern, col, re.IGNORECASE):
                    logging.info(f"Column '{col}' in dataset '{key}' skipped (contains 'id').")
                    continue

                # ตรวจสอบเฉพาะคอลัมน์ที่เป็น object (string)
                if df[col].dtype != "object":
                    continue

                # ถ้าไม่มีตัวเลขเลย ให้ข้าม
                if not df[col].str.contains(r"\d", na=False).any():
                    continue

                # -----------------------------
                # 1. แปลงเป็น datetime
                # -----------------------------
                try:
                    datetime_series = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format=date_format)
                except Exception as e:
                    logging.error(f"Error parsing datetime in column '{col}' of dataset '{key}': {e}")
                    datetime_series = pd.Series([pd.NaT] * len(df[col]))

                non_na_ratio = datetime_series.notna().mean()
                logging.info(f"Column '{col}' in dataset '{key}' datetime conversion ratio: {non_na_ratio:.2f}")

                # หากอัตราส่วนไม่ถึง threshold ใช้ fallback ด้วย dateutil.parser
                if non_na_ratio < threshold:
                    def safe_parse(x):
                        if not isinstance(x, str):
                            return pd.NaT
                        try:
                            return parse(x)
                        except Exception:
                            return pd.NaT

                    datetime_series = df[col].apply(safe_parse)
                    non_na_ratio = datetime_series.notna().mean()
                    logging.info(f"Column '{col}' in dataset '{key}' fallback datetime conversion ratio: {non_na_ratio:.2f}")

                #  หากแปลง datetime สำเร็จ ให้แทนที่คอลัมน์
                if non_na_ratio >= threshold:
                    df[col] = datetime_series
                    logging.info(f"Column '{col}' in dataset '{key}' converted to datetime.")
                    continue

                # -----------------------------
                # 2. แปลงเป็น numeric
                # -----------------------------
                cleaned = df[col].str.replace(r"[^\d\.-]", "", regex=True)  # ลบอักขระที่ไม่ใช่ตัวเลข จุด หรือ "-"
                numeric_series = pd.to_numeric(cleaned, errors="coerce")
                non_na_ratio = numeric_series.notna().mean()
                logging.info(f"Column '{col}' in dataset '{key}' numeric conversion ratio: {non_na_ratio:.2f}")

                if non_na_ratio >= threshold:
                    df[col] = numeric_series
                    logging.info(f"Column '{col}' in dataset '{key}' converted to numeric.")
                else:
                    logging.info(f"Column '{col}' in dataset '{key}' conversion skipped (ratio below threshold).")

        logging.info("Preprocessing complete.")


    def get_data(self, key: str) -> pd.DataFrame:
        """
        ดึงข้อมูล DataFrame ที่โหลดมาแล้วออกมาตาม key ที่ระบุ
        Parameters:
            key: ตัวระบุของ dataset (เช่น "df1", "df2")
        Returns:
            DataFrame ที่โหลดมาแล้วจาก self._data
        """
        # ตรวจสอบว่า key ที่ระบุมีอยู่ใน self._data หรือไม่
        if key not in self._data:
            raise ValueError(f"Data for key '{key}' not loaded.")
        return self._data[key]  # คืนค่า DataFrame ที่เก็บไว้ใน self._data สำหรับ key นั้น
