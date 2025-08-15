# ip_app.py
import streamlit as st
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from docx import Document
from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize
import requests
from typing import Dict, List
import base64
from weasyprint import HTML
import markdown2
import re
import socket
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bridge-engineering-app")

# ================== 获取本机局域网 IP ==================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"获取IP失败: {str(e)}")
        return "无法获取IP"

# ================== 获取当前城市 ==================
def get_location():
    try:
        response = requests.get("https://ipapi.co/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get("city")
            region = data.get("region")
            country = data.get("country_name")
            lat = data.get("latitude")
            lon = data.get("longitude")
            if city and lat is not None and lon is not None:
                return city, lat, lon
            location_name = city or region or country or "未知位置"
            return location_name, lat, lon
    except Exception as e:
        logger.warning(f"获取位置失败: {str(e)}")
        st.warning(f"⚠️ 获取位置失败: {str(e)}")
    return None, None, None

# ================== 获取详细的天气摘要 ==================
def get_weather_summary(weather_data: dict) -> str:
    """生成详细的天气摘要，用于提供给LLM更丰富的信息"""
    if not weather_data:
        return "无法获取天气数据"
    # 获取今日数据
    today_max_temp = weather_data["daily"]["temperature_2m_max"][0]
    today_min_temp = weather_data["daily"]["temperature_2m_min"][0]
    today_precip = weather_data["daily"]["precipitation_sum"][0]
    today_wind_max = weather_data["daily"]["wind_speed_10m_max"][0]
    # 获取未来24小时的关键数据
    hourly = weather_data["hourly"]
    current_temp = hourly["temperature_2m"][0]
    current_humidity = hourly["relative_humidity_2m"][0]
    current_wind = hourly["wind_speed_10m"][0]
    # 分析降水趋势
    precipitation = hourly["precipitation"][:24]
    rain_hours = sum(1 for p in precipitation if p > 0.1)
    max_precip = max(precipitation)
    # 分析温度趋势
    temps = hourly["temperature_2m"][:24]
    temp_trend = "上升" if temps[-1] > temps[0] else "下降" if temps[-1] < temps[0] else "平稳"
    # 生成详细的天气摘要
    summary = f"📍 当前位置天气概况:\n"
    summary += f"• 实时温度: {current_temp}°C, 湿度: {current_humidity}%, 风速: {current_wind}m/s\n"
    summary += f"• 今日温度范围: {today_min_temp}°C - {today_max_temp}°C\n"
    if today_precip > 0:
        summary += f"• 今日降水: {today_precip}mm, 预计有{rain_hours}小时降雨，最大降雨量: {max_precip}mm/h\n"
    else:
        summary += "• 今日无降水\n"
    summary += f"• 今日最高风速: {today_wind_max}m/s, 24小时温度趋势: {temp_trend}\n"
    # 添加可能的施工影响分析
    impacts = []
    if today_wind_max > 15:
        impacts.append("⚠️ 大风(>15m/s)可能影响高空作业安全，需加固临时结构")
    if today_wind_max > 10:
        impacts.append("⚠️ 风速较高(>10m/s)，需注意吊装作业安全")
    if today_precip > 5:
        impacts.append("⚠️ 降雨量较大(>5mm)，可能影响混凝土浇筑质量")
    if today_precip > 0.5:
        impacts.append("⚠️ 有降雨，需准备防雨措施")
    if current_temp > 35:
        impacts.append("⚠️ 高温(>35°C)可能加速混凝土水分蒸发，需加强养护")
    if current_temp < 5:
        impacts.append("⚠️ 低温(<5°C)可能影响混凝土凝固，需采取保温措施")
    if current_humidity < 40:
        impacts.append("⚠️ 低湿度(<40%)可能加速混凝土表面水分蒸发")
    if impacts:
        summary += "📌 施工影响提示:\n" + "\n".join(impacts)
    else:
        summary += "✅ 当前天气条件适宜桥梁施工"
    return summary

# ================== 获取天气数据 ==================
def get_weather(lat: float, lon: float):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
            "timezone": "Asia/Shanghai",
            "forecast_days": 7
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"天气API错误: {response.status_code}")
            st.warning(f"⚠️ 天气API错误: {response.status_code}")
    except Exception as e:
        logger.warning(f"获取天气失败: {str(e)}")
        st.warning(f"⚠️ 获取天气失败: {str(e)}")
    return None

# ================== 绘制天气图表 ==================
def plot_weather(weather_data: dict):
    hourly = weather_data["hourly"]
    df_hourly = pd.DataFrame({
        "时间": pd.to_datetime(hourly["time"]),
        "温度(°C)": hourly["temperature_2m"],
        "湿度(%)": hourly["relative_humidity_2m"],
        "风速(m/s)": hourly["wind_speed_10m"]
    }).set_index("时间")
    daily = weather_data["daily"]
    df_daily = pd.DataFrame({
        "日期": pd.to_datetime(daily["time"]),
        "最高温(°C)": daily["temperature_2m_max"],
        "最低温(°C)": daily["temperature_2m_min"],
        "降水(mm)": daily["precipitation_sum"],
        "最大风速(m/s)": daily["wind_speed_10m_max"]
    }).set_index("日期")
    st.subheader("🌤️ 未来7天天气预报")
    st.markdown("### 按小时")
    st.line_chart(df_hourly[["温度(°C)", "湿度(%)"]].iloc[:48])  # 显示前48小时
    st.line_chart(df_hourly[["风速(m/s)"]].iloc[:48])
    st.markdown("### 按天")
    st.dataframe(df_daily.style.format({
        "最高温(°C)": "{:.1f}",
        "最低温(°C)": "{:.1f}",
        "降水(mm)": "{:.1f}",
        "最大风速(m/s)": "{:.1f}"
    }))
    # 显示详细的天气摘要
    weather_summary = get_weather_summary(weather_data)
    with st.expander("📊 详细的天气施工影响分析"):
        st.markdown(weather_summary)
    return df_daily

# ================== 配置 ==================
# 使用 st.secrets —— 必须创建 .streamlit/secrets.toml
MODEL_MODE = st.secrets.get("MODEL_MODE", "api")  # 默认使用API模式，避免用户需要本地模型
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = st.secrets.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").strip()
LLM_MODEL = st.secrets.get("LLM_MODEL", "deepseek-chat")

# 本地模型路径 - 云部署中通常不使用
MODEL_PATH = "/app/models/text2vec-base-chinese"
CHROMA_PATH = "/app/chroma_data"
DATA_DIR = "/app/data/docs"
CACHE_FOLDER = "/app/.cache"
LOCALAI_URL = "http://localhost:8082"

# 报告输出目录
REPORT_OUTPUT_DIR = "/app/reports"

# 创建必要目录
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# 风险阈值
TEMPERATURE_THRESHOLD = 60   # °C
CRACK_WIDTH_THRESHOLD = 0.15 # mm
STRAIN_THRESHOLD = 120       # με

# ================== 设置页面配置 ==================
st.set_page_config(
    page_title="桥梁工程AI助手",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 自定义CSS样式 ==================
def load_css():
    st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #1f77b4; color: white; border-radius: 8px;
        height: 40px; font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #145a8c; transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div { background-color: #1f77b4; }
    .stAlert { border-radius: 8px; }
    .st-expanderHeader { font-weight: 600; color: #1f77b4; }
    .tab-content {
        padding: 1rem; border-radius: 8px;
        background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .file-card {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem;
        margin-bottom: 1rem; background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.3s ease;
    }
    .file-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); transform: translateY(-2px);
    }
    .status-badge {
        padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }
    .status-processed { background-color: #d4edda; color: #155724; }
    .status-pending { background-color: #fff3cd; color: #856404; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .metric-card {
        background: linear-gradient(135deg, #1f77b4 0%, #4e79a7 100%);
        color: white; border-radius: 10px; padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }
    .metric-label { font-size: 1rem; opacity: 0.9; }
    .ai-response, .report-preview {
        background-color: #e9f7fe; border-left: 4px solid #1f77b4;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    }
    .system-message {
        background-color: #f8f9fa; border-radius: 8px; padding: 1rem; margin: 1rem 0;
    }
    .risk-high { color: #d9534f; font-weight: bold; }
    .risk-medium { color: #f0ad4e; font-weight: bold; }
    .risk-low { color: #5cb85c; font-weight: bold; }
    .model-selector {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .network-tip {
        font-size: 0.9rem; color: #666; text-align: center;
        margin-top: 0.5rem;
    }
    .weather-summary {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ================== 模型调用函数 ==================
def call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """统一的模型调用接口"""
    if st.session_state.model_mode == "api":
        if not DEEPSEEK_API_KEY:
            return "❌ 未配置 DEEPSEEK_API_KEY，请联系管理员"
        return call_deepseek_api(prompt, max_tokens, temperature)
    else:
        return call_local_model(prompt, max_tokens, temperature)

def call_deepseek_api(prompt: str, max_tokens: int, temperature: float) -> str:
    """调用DeepSeek API"""
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=90
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            return re.sub(r'^```markdown|```$', '', content, flags=re.MULTILINE).strip()
        return f"❌ API错误: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ API调用失败: {str(e)}"

def call_local_model(prompt: str, max_tokens: int, temperature: float) -> str:
    """调用本地模型 - 云部署中通常不使用"""
    return "❌ 本地模型模式在云部署中不可用，请使用API模式"

# ================== 初始化会话状态 ==================
def init_session_state():
    state_keys = {
        'uploaded_files': {},
        'progress': 0,
        'status': "系统就绪",
        'system_status': "系统就绪",
        'knowledge_count': 0,
        'ai_response': "",
        'ai_status': "等待查询",
        'last_quality_report': "",
        'last_emergency_plan': "",
        'last_report_paths': {},
        'current_tab': "文档管理",
        'field_data': None,
        'model_mode': "api",  # 默认使用API模式
        'weather_data': None,
        'location': None
    }
    for k, v in state_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ================== AI问答 ==================
def ai_query(query):
    if not query.strip():
        st.warning("⚠️ 请输入问题")
        return
    st.session_state.ai_status = "🔄 正在思考..."
    st.session_state.ai_response = ""
    try:
        # 云部署中禁用知识库检索，简化功能
        context_str = "桥梁工程专业知识库"
        
        # 获取天气信息用于回答
        weather_info = "无天气信息"
        if st.session_state.weather_data:
            weather_info = get_weather_summary(st.session_state.weather_data)
        
        prompt = f"""任务：{query}
资料：{context_str}
当前天气情况：{weather_info}
要求：专业回答，30-80字，直接给出风险点和措施，考虑当前天气条件的影响。"""
        
        st.session_state.ai_response = call_llm(prompt, max_tokens=150, temperature=0.1)
        st.session_state.ai_status = "✅ 回答完成"
    except Exception as e:
        st.session_state.ai_response = f"❌ 失败: {str(e)}"
        st.session_state.ai_status = "❌ 处理失败"

# ================== 风险评估 ==================
def assess_risk_level(field_data: Dict) -> Dict:
    alerts = []
    s = field_data.get("sensor_data", {})
    temp = s.get("concrete_temp", 0)
    crack = s.get("crack_width", 0)
    strain = s.get("strain", 0)
    if temp > TEMPERATURE_THRESHOLD:
        alerts.append(f"混凝土温度 {temp}°C > {TEMPERATURE_THRESHOLD}°C")
    if crack > CRACK_WIDTH_THRESHOLD:
        alerts.append(f"裂缝宽度 {crack}mm > {CRACK_WIDTH_THRESHOLD}mm")
    if strain > STRAIN_THRESHOLD:
        alerts.append(f"应变值 {strain}με > {STRAIN_THRESHOLD}με")
    level = "高" if len(alerts) >= 2 else "中" if alerts else "低"
    return {"risk_level": level, "alerts": alerts, "has_risk": len(alerts) > 0}

# ================== 报告生成 ==================
def generate_technical_report(field_data: Dict, report_type: str):
    # 云部署中简化，不依赖知识库
    risk_result = assess_risk_level(field_data)
    
    # 使用详细的天气信息
    weather_info = field_data.get("weather_info", "无法获取天气数据")
    if "无法获取" in weather_info and st.session_state.weather_data:
        weather_info = get_weather_summary(st.session_state.weather_data)
    
    if report_type == "quality_monitoring":
        prompt = f"""
你是一位桥梁工程专家，请生成《质量监测报告》。
【工程信息】
- 阶段：{field_data.get('project_phase', '未知')}
- 位置：{field_data.get('location', '未知')}
- 日期：{field_data.get('date', '未知')}
- 天气：{field_data.get('weather', '未知')}
【详细天气信息】
{weather_info}
【监测数据】
- 混凝土温度：{field_data['sensor_data'].get('concrete_temp', 'N/A')}°C
- 应变值：{field_data['sensor_data'].get('strain', 'N/A')}με
- 裂缝宽度：{field_data['sensor_data'].get('crack_width', 'N/A')}mm
【现场备注】
{field_data.get('inspection_notes', '无')}
【要求】
- 使用 Markdown
- 包含：工程概况、数据汇总、异常分析、处理建议
- 语言专业，不少于200字
- 结合详细天气信息对施工的影响提出具体建议
- 分析温度、湿度、风速、降水等因素对当前施工环节的具体影响
        """
    elif report_type == "risk_emergency_plan":
        if not risk_result["has_risk"]: 
            return "当前无显著风险，无需生成应急预案。"
        
        prompt = f"""
你是一位桥梁工程应急指挥专家，请生成《突发事件应急处置预案》。
【事件背景】
- 工程阶段：{field_data.get('project_phase', '未知')}
- 位置：{field_data.get('location', '未知')}
- 时间：{field_data.get('date', '未知')}
【详细天气信息】
{weather_info}
【风险评估】
风险等级：{risk_result['risk_level']}级
异常项：
{chr(10).join(['- ' + a for a in risk_result['alerts']])}
【要求】
- 标题为"关于{field_data.get('location', '某桥梁')}的应急处置预案"
- 包含：事件描述、响应级别、组织架构、处置措施
- 措施必须具体可执行，考虑当前天气条件下的特殊要求
- 针对温度、湿度、风速、降水等天气因素制定相应的应对方案
- 明确各环节责任人和时间节点
- 输出为 Markdown
        """
    else:
        return "未知类型"
    
    return call_llm(prompt, max_tokens=1000, temperature=0.3)

# ================== 保存和转换报告 ==================
def save_report_to_file(content: str, report_type: str, location: str = "", format_type: str = "md"):
    """
    保存报告到文件，并根据 format_type 转换为不同格式。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{report_type}_{timestamp}"
    if location:
        base_filename += f"_{location}"
    
    # 创建一个临时的 Markdown 文件路径
    md_path = os.path.join(REPORT_OUTPUT_DIR, f"{base_filename}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- 生成时间: {timestamp} -->\n")
        f.write(content)
    
    # 云部署中只支持Markdown格式
    if format_type == "docx" or format_type == "pdf":
        return md_path  # 只返回Markdown文件
    else:
        return md_path

# ================== 主应用 ==================
def main():
    init_session_state()
    load_css()
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f77b4 0%, #4e79a7 100%); 
                border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0; font-size: 2.5rem;">🌉 桥梁工程AI助手</h1>
        <p style="opacity: 0.9; font-size: 1.1rem;">知识管理 · 智能问答 · 报告生成</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ 天气数据获取 ============
    if st.session_state.weather_data is None:
        with st.spinner("正在获取当前位置和天气数据..."):
            city, lat, lon = get_location()
            if city and lat is not None and lon is not None:
                st.session_state.location = city
                st.success(f"📍 确定位置: {city} | 获取经纬度: {lat:.4f}, {lon:.4f}")
                weather = get_weather(lat, lon)
                if weather:
                    st.session_state.weather_data = weather
                    st.success(f"🌤️ 天气数据获取成功")
                else:
                    st.warning("无法获取天气数据。")
            else:
                st.warning("无法确定您的位置，请检查网络连接或手动输入位置信息。")
    
    # 显示天气
    if st.session_state.weather_data:
        plot_weather(st.session_state.weather_data)
    else:
        st.warning("天气数据不可用，请检查网络连接")
    
    # 模型选择器 - 云部署中只允许API模式
    with st.sidebar:
        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        st.subheader("模型设置")
        st.info("云部署版本仅支持API模式")
        st.success("✅ API模式已启用")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="system-message">
        <b>系统状态</b>: {st.session_state.system_status} | 
        <b>知识片段</b>: {st.session_state.knowledge_count} | 
        <b>模型</b>: DeepSeek API
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 文档管理", "🤖 AI问答", "📊 报告生成", "⚙️ 系统状态"])
    
    # ============= 文档管理 =============
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📁 文档上传")
        st.warning("云部署版本暂不支持文档上传和知识库构建功能")
        st.info("如需使用完整功能，请联系管理员获取本地部署版本")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============= AI问答 =============
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("❓ 智能问答")
        query = st.text_area("输入问题", height=100, placeholder="斜拉桥主塔施工风险？")
        examples = [
            "混凝土温控措施？", "斜拉索防腐方法？", "主梁架设安全要点？"
        ]
        cols = st.columns(2)
        for i, q in enumerate(examples):
            with cols[i % 2]:
                if st.button(q, key=f"q{i}"): 
                    query = q
        if st.button("🔍 获取回答"):
            ai_query(query)
        st.markdown(f"**状态**: {st.session_state.ai_status}")
        if st.session_state.ai_response:
            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
            st.markdown("### 💡 回答")
            st.write(st.session_state.ai_response)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============= 报告生成 =============
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📊 报告生成")
        method = st.radio("输入方式", ["📄 手动"], horizontal=True)
        
        # 数据输入部分 - 只保留手动输入
        if method == "📄 手动":
            with st.form("manual_form"):
                project_phase = st.text_input("工程阶段", "主塔施工")
                location = st.text_input("位置", st.session_state.location or "施工现场")
                date = st.text_input("日期", datetime.now().strftime("%Y-%m-%d"))
                weather = st.text_input("天气", "晴")
                
                # 生成详细的天气摘要用于显示
                weather_info = "无法获取天气数据"
                if st.session_state.weather_data:
                    weather_info = get_weather_summary(st.session_state.weather_data)
                    st.markdown(f'<div class="weather-summary">{weather_info}</div>', unsafe_allow_html=True)
                
                inspection_notes = st.text_area("现场备注", "混凝土表面有细微裂缝，需加强养护")
                temp = st.number_input("混凝土温度", value=62.0, min_value=0.0)
                crack = st.number_input("裂缝宽度", value=0.18, min_value=0.0, step=0.01)
                strain = st.number_input("应变值", value=130.0, min_value=0.0, step=1.0)
                
                if st.form_submit_button("保存数据"):
                    st.session_state.field_data = {
                        "project_phase": project_phase,
                        "location": location,
                        "date": date,
                        "weather": weather,
                        "weather_info": weather_info,
                        "inspection_notes": inspection_notes,
                        "sensor_data": {
                            "concrete_temp": temp, 
                            "crack_width": crack, 
                            "strain": strain
                        }
                    }
                    st.success("数据已保存！")
        
        # 显示当前数据
        if st.session_state.field_data:
            with st.expander("当前数据预览"):
                st.json(st.session_state.field_data)
        
        # 风险评估
        if st.session_state.field_data:
            risk = assess_risk_level(st.session_state.field_data)
            level_class = f"risk-{ 'high' if risk['risk_level']=='高' else 'medium' if risk['risk_level']=='中' else 'low' }"
            st.markdown(f"**风险等级**: <span class='{level_class}'>{risk['risk_level']}风险</span>", unsafe_allow_html=True)
            if risk['alerts']:
                for a in risk['alerts']: 
                    st.error(f"⚠️ {a}")
            else:
                st.success("✅ 未检测到风险")
        else:
            st.warning("⚠️ 请先输入数据")
        
        # 报告生成按钮
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈 生成质量报告", use_container_width=True, key="gen_quality"):
                if st.session_state.field_data:
                    with st.spinner("生成质量报告中..."):
                        rep = generate_technical_report(st.session_state.field_data, "quality_monitoring")
                        if rep and "❌" not in rep:
                            st.session_state.last_quality_report = rep
                            # 保存为Markdown格式
                            st.session_state.last_report_paths = {
                                "md": save_report_to_file(rep, "quality_report", st.session_state.field_data.get('location', ''), "md")
                            }
                            st.success("质量报告生成完成！")
                        else:
                            st.error(f"报告生成失败: {rep}")
                else:
                    st.warning("请先输入数据")
        
        with c2:
            if st.button("🚨 生成应急预案", use_container_width=True, key="gen_emergency"):
                if st.session_state.field_data:
                    risk = assess_risk_level(st.session_state.field_data)
                    if risk["has_risk"]:
                        with st.spinner("生成应急预案中..."):
                            plan = generate_technical_report(st.session_state.field_data, "risk_emergency_plan")
                            if plan and "❌" not in plan and "无需生成" not in plan:
                                st.session_state.last_emergency_plan = plan
                                # 保存为Markdown格式
                                st.session_state.last_report_paths = {
                                    "md": save_report_to_file(plan, "emergency_plan", st.session_state.field_data.get('location', ''), "md")
                                }
                                st.success("应急预案生成完成！")
                            else:
                                st.warning("应急预案未生成: " + plan)
                    else:
                        st.info("当前无风险，无法生成应急预案")
                else:
                    st.warning("请先输入数据")
        
        # 报告预览和下载 - 仅支持Markdown
        st.markdown("### 📄 质量报告预览")
        if st.session_state.last_quality_report:
            st.markdown(st.session_state.last_quality_report)
            # 只提供Markdown下载
            st.download_button(
                label="📥 下载 Markdown (.md)",
                data=st.session_state.last_quality_report,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("暂无质量报告，请填写数据并点击'生成质量报告'按钮")
        
        st.markdown("### 📄 应急预案预览")
        if st.session_state.last_emergency_plan:
            st.markdown(st.session_state.last_emergency_plan)
            # 只提供Markdown下载
            st.download_button(
                label="📥 下载 Markdown (.md)",
                data=st.session_state.last_emergency_plan,
                file_name=f"emergency_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("暂无应急预案，当检测到风险时可生成")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============= 系统状态 =============
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📊 系统概览")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">知识片段</div><div class="metric-value">0</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card" style="background: #2ca02c;"><div class="metric-label">上传文件</div><div class="metric-value">0</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card" style="background: #d62728;"><div class="metric-label">报告数量</div><div class="metric-value">-</div></div>', unsafe_allow_html=True)
        
        st.subheader("ℹ️ 云部署说明")
        st.markdown("""
        - 本版本为简化云部署版本，仅提供核心AI问答和报告生成功能
        - 不支持文档上传和知识库构建功能
        - 使用DeepSeek API提供AI能力，无需本地模型
        - 如需完整功能，请联系管理员获取本地部署版本
        - 任何问题请发送邮件至: support@bridge-ai.com
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666; border-top: 1px solid #eee; margin-top: 2rem;">
        © 2025 桥梁工程AI助手 | 数据安全处理，专业可靠
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
