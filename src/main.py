import torch
from transformers import Qwen2VLForConditionalGeneration, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain.agents import initialize_agent, Tool, AgentType
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import time
from langchain.llms.base import LLM
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
import threading

# --- Класс кастомной LLM ---
class CustomLLM(LLM):
    model_name: str = "Qwen-8B"
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()

    @property
    def _llm_type(self) -> str:
        return "custom_qwen_8b"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]

# --- Глобальный Excel Writer (потокобезопасный) ---
class ExcelWriter:
    def __init__(self, output_file='output.xlsx'):
        self.output_file = output_file
        self.lock = threading.Lock()
        self.data = []

    def save(self, vin, url, summary):
        with self.lock:
            self.data.append({'VIN': vin, 'URL': url, 'Summary': summary})
            pd.DataFrame(self.data).to_excel(self.output_file, index=False)

# --- Инструменты ---
def yandex_search_tool(vin: str) -> list:
    ua = UserAgent()
    options = Options()
    options.add_argument(f'user-agent={ua.random}')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    full_html = ""
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        for page in range(2):
            search_url = f"https://yandex.ru/search/?text={vin}&p={page}"
            driver.get(search_url)
            time.sleep(3)
            html = driver.page_source
            full_html += f"\n<!-- PAGE {page + 1} -->\n{html}"
    except Exception as e:
        print(f"[Ошибка при поиске VIN {vin}]: {e}")
    finally:
        try:
            driver.quit()
        except:
            pass
    soup = BeautifulSoup(full_html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('http') and 'yandex' not in href:
            links.append(href)
        if len(links) >= 10:
            break
    return links

def get_page_text_tool(url: str) -> str:
    try:
        headers = {'User-Agent': UserAgent().random}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for s in soup(['script', 'style']):
            s.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"[Ошибка при загрузке {url}]: {e}")
        return ""

def save_to_excel_tool(args: dict, excel_writer=None) -> str:
    if not isinstance(args, dict):
        return "Ошибка: SaveToExcelTool ожидает словарь с ключами 'vin', 'url', 'summary'."
    vin = args.get('vin')
    url = args.get('url')
    summary = args.get('summary')
    if excel_writer is not None:
        excel_writer.save(vin, url, summary)
        return f"Сохранено: VIN={vin}, URL={url}"
    else:
        return "ExcelWriter не передан"

# --- Основная функция для запуска пайплайна из Jupyter ---
def run_vin_pipeline(vin_list, llm, output_file='output.xlsx'):
    """
    vin_list: список VIN-номеров
    llm: объект CustomLLM (уже загруженный)
    output_file: имя выходного Excel-файла
    """
    excel_writer = ExcelWriter(output_file=output_file)

    # Обёртки для инструментов с excel_writer
    def save_to_excel_tool_wrapped(args):
        return save_to_excel_tool(args, excel_writer=excel_writer)

    tools = [
        Tool(
            name="YandexSearchTool",
            func=yandex_search_tool,
            description="Ищет VIN в Яндексе и возвращает список ссылок (до 10). На вход принимает VIN-номер (строка)."
        ),
        Tool(
            name="GetPageTextTool",
            func=get_page_text_tool,
            description="Получает текст страницы по URL. На вход принимает URL (строка)."
        ),
        Tool(
            name="SaveToExcelTool",
            func=save_to_excel_tool_wrapped,
            description="Сохраняет найденный VIN, ссылку и summary в Excel. На вход принимает словарь с ключами 'vin', 'url', 'summary'."
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    for vin in vin_list:
        prompt = (
            f"Для VIN-номера {vin}:\n"
            f"1. Вызови YandexSearchTool, чтобы получить список ссылок.\n"
            f"2. Для каждой ссылки вызови GetPageTextTool, чтобы получить текст.\n"
            f"3. Проанализируй текст: если VIN найден, сделай краткое summary (фрагмент с VIN) и вызови SaveToExcelTool с vin, url и summary.\n"
            f"Если VIN не найден ни на одной странице — ничего не сохраняй."
        )
        agent.run(prompt)
    print(f"Готово! Результаты сохранены в {output_file}")

# --- Пример вызова (раскомментируйте для запуска в Jupyter) ---
# llm = CustomLLM(model_name_or_path='Qwen3-8B')
# vin_list = ['RH21J-001845', 'SALCP2BG2HH699250', '1HD4CR21XDC451359', 'ZDMAA06JAHB019322']
# run_vin_pipeline(vin_list, llm, output_file='output.xlsx')