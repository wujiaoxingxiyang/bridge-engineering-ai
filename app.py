# app.py
import streamlit as st
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import requests
from typing import Dict, List
import re

# ================== 获取本机局域网 IP ==================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
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
        st.warning(f"获取位置失败: {str(e)}")
    return None, None, None

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
            st.warning(f"天气API错误: {response.status_code}")
    except Exception as e:
        st.warning(f"获取天气失败: {str(e)}")
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
    st.line_chart(df_hourly[["温度(°C)", "湿度(%)"]].iloc[:48])
    st.line_chart(df_hourly[["风速(m/s)"]].iloc[:48])
    st.markdown("### 按天")
    st.dataframe(df_daily.style.format({
        "最高温(°C)": "{:.1f}",
        "最低温(°C)": "{:.1f}",
        "降水(mm)": "{:.1f}",
        "最大风速(m/s)": "{:.1f}"
    }))

# ================== 配置 ==================
# 使用 st.secrets —— 在Hugging Face Secrets中设置
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
LLM_MODEL = st.secrets.get("LLM_MODEL", "deepseek-chat")

# 报告输出目录
REPORT_OUTPUT_DIR = "reports"
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

# ================== 模型调用函数 ==================
def call_llm(prompt: str, max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """统一的模型调用接口"""
    if not DEEPSEEK_API_KEY:
        return "❌ 未配置 DEEPSEEK_API_KEY，请联系管理员"
    
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
            "https://api.deepseek.com/v1/chat/completions",
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
    risk_result = assess_risk_level(field_data)
    
    # 使用详细的天气信息
    weather_info = "无法获取天气数据"
    if "weather_info" in field_data:
        weather_info = field_data["weather_info"]
    
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
- 明确各环节责任人和时间节点
- 输出为 Markdown
        """
    else:
        return "未知类型"
    
    return call_llm(prompt, max_tokens=1000, temperature=0.3)

# ================== 主应用 ==================
def main():
    # 初始化会话状态
    if 'field_data' not in st.session_state:
        st.session_state.field_data = None
    if 'last_quality_report' not in st.session_state:
        st.session_state.last_quality_report = ""
    if 'last_emergency_plan' not in st.session_state:
        st.session_state.last_emergency_plan = ""
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1f77b4 0%, #4e79a7 100%); 
                border-radius: 10px; margin-bottom: 2rem; color: white;">
        <h1 style="margin: 0; font-size: 2.5rem;">🌉 桥梁工程AI助手</h1>
        <p style="opacity: 0.9; font-size: 1.1rem;">知识管理 · 智能问答 · 报告生成</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============ 天气数据获取 ============
    if 'weather_data' not in st.session_state:
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
    if 'weather_data' in st.session_state:
        plot_weather(st.session_state.weather_data)
    else:
        st.warning("天气数据不可用，请检查网络连接")
    
    # 模型状态提示
    with st.sidebar:
        st.subheader("模型设置")
        if DEEPSEEK_API_KEY:
            st.success("✅ API模式已启用")
        else:
            st.warning("⚠️ API密钥未配置")
    
    tab1, tab2, tab3 = st.tabs(["🤖 AI问答", "📊 报告生成", "⚙️ 系统状态"])
    
    # ============= AI问答 =============
    with tab1:
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
            if not query.strip():
                st.warning("⚠️ 请输入问题")
            else:
                with st.spinner("正在思考..."):
                    # 简化AI问答，不依赖知识库
                    prompt = f"""任务：{query}
要求：专业回答，30-80字，直接给出风险点和措施。"""
                    response = call_llm(prompt, max_tokens=150, temperature=0.1)
                    st.markdown("### 💡 回答")
                    st.write(response)
    
    # ============= 报告生成 =============
    with tab2:
        st.subheader("📊 报告生成")
        method = st.radio("输入方式", ["📄 手动"], horizontal=True)
        
        # 数据输入部分 - 只保留手动输入
        if method == "📄 手动":
            with st.form("manual_form"):
                project_phase = st.text_input("工程阶段", "主塔施工")
                location = st.text_input("位置", st.session_state.get('location', '施工现场'))
                date = st.text_input("日期", datetime.now().strftime("%Y-%m-%d"))
                weather = st.text_input("天气", "晴")
                
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
            level_color = "red" if risk['risk_level'] == '高' else "orange" if risk['risk_level'] == '中' else "green"
            st.markdown(f"**风险等级**: <span style='color:{level_color};font-weight:bold;'>{risk['risk_level']}风险</span>", unsafe_allow_html=True)
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
            if st.button("📈 生成质量报告", use_container_width=True):
                if st.session_state.field_data:
                    with st.spinner("生成质量报告中..."):
                        rep = generate_technical_report(st.session_state.field_data, "quality_monitoring")
                        if rep and "❌" not in rep:
                            st.session_state.last_quality_report = rep
                            st.success("质量报告生成完成！")
                        else:
                            st.error(f"报告生成失败: {rep}")
                else:
                    st.warning("请先输入数据")
        
        with c2:
            if st.button("🚨 生成应急预案", use_container_width=True):
                if st.session_state.field_data:
                    risk = assess_risk_level(st.session_state.field_data)
                    if risk["has_risk"]:
                        with st.spinner("生成应急预案中..."):
                            plan = generate_technical_report(st.session_state.field_data, "risk_emergency_plan")
                            if plan and "❌" not in plan and "无需生成" not in plan:
                                st.session_state.last_emergency_plan = plan
                                st.success("应急预案生成完成！")
                            else:
                                st.warning("应急预案未生成: " + plan)
                    else:
                        st.info("当前无风险，无法生成应急预案")
                else:
                    st.warning("请先输入数据")
        
        # 报告预览和下载
        st.markdown("### 📄 质量报告预览")
        if st.session_state.last_quality_report:
            st.markdown(st.session_state.last_quality_report)
            st.download_button(
                label="📥 下载 Markdown",
                data=st.session_state.last_quality_report,
                file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("暂无质量报告，请填写数据并点击'生成质量报告'按钮")
        
        st.markdown("### 📄 应急预案预览")
        if st.session_state.last_emergency_plan:
            st.markdown(st.session_state.last_emergency_plan)
            st.download_button(
                label="📥 下载 Markdown",
                data=st.session_state.last_emergency_plan,
                file_name=f"emergency_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("暂无应急预案，当检测到风险时可生成")
    
    # ============= 系统状态 =============
    with tab3:
        st.subheader("📊 系统概览")
        st.markdown("""
        - 本版本为简化云部署版本，仅提供核心AI问答和报告生成功能
        - 不支持文档上传和知识库构建功能
        - 使用DeepSeek API提供AI能力
        - 如需完整功能，请联系管理员
        """)
        
        st.subheader("ℹ️ 使用说明")
        st.markdown("""
        1. 在"报告生成"标签页输入工程数据
        2. 点击"保存数据"按钮
        3. 查看风险评估结果
        4. 生成相应的报告
        5. 下载报告文件
        """)

if __name__ == "__main__":
    main()
