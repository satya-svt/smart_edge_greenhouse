"""
app.py
======
Interactive Web/Mobile Dashboard & Control Center for Smart Greenhouse Monitoring.
Based on IEEE Access Paper:
"Edge-AI Based Smart Greenhouse Monitoring and Automated Plant Growth Control Using Raspberry Pi"

Implements:
- Section III Experimental Condition Injection (Presets)
- Real-time Virtual Raspberry Pi GPIO BCM Actuator Control (Relays: 17, 27, 22)
- Edge-AI Diagnostics (Classification Confidence, Probabilities, Edge vs Cloud Latency)
- Closed-Loop Digital Twin / HIL Simulation Feedback
- Plotly Multi-Parameter Telemetry & Actuator Digital Waveform Charts
- Research Experiment Logging and CSV Export
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hardware_hal import (
    virtual_gpio,
    GPIO,
    IS_HARDWARE_EMULATED,
    PIN_WATER_PUMP,
    PIN_COOLING_FAN,
    PIN_GROW_LIGHT,
    EXPERIMENTAL_PRESETS,
    VirtualSensorArray,
    initialize_hardware,
    HIGH,
    LOW,
)
from edge_ai_engine import (
    CROP_PROFILES,
    CONDITION_CLASSES,
    GreenhouseEdgeAI,
    DEFAULT_MODEL_PATH,
    train_and_save_model,
)

# ============================================================================
# STREAMLIT PAGE CONFIGURATION & THEME
# ============================================================================
st.set_page_config(
    page_title="Edge-AI Smart Greenhouse | Raspberry Pi Digital Twin",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Dark Cyber-Agri CSS Styling
st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        background-color: #0b111e;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #132238 0%, #1e3a5f 100%);
        border: 1px solid #2d4d76;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Actuator Cards */
    .actuator-card {
        background-color: #111a2e;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #1e2d4d;
        margin-bottom: 12px;
        transition: all 0.25s ease;
    }
    .actuator-card:hover {
        border-color: #3b82f6;
    }
    
    .status-pill-on {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.35);
    }
    
    .status-pill-off {
        background-color: rgba(148, 163, 184, 0.1);
        color: #94a3b8;
        border: 1px solid #475569;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Condition Badges */
    .badge-optimal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #059669;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
    }
    
    /* Benchmark comparison block */
    .benchmark-card {
        background-color: #0f172a;
        border: 1px solid #223254;
        border-radius: 10px;
        padding: 14px 16px;
    }
    
    /* Metric Card Customization */
    div[data-testid="stMetric"] {
        background-color: #111a2e;
        border: 1px solid #1e2d4d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# STATE INITIALIZATION
# ============================================================================
if "initialized" not in st.session_state:
    initialize_hardware(virtual_gpio)
    st.session_state.gpio = virtual_gpio
    st.session_state.sensor_array = VirtualSensorArray(virtual_gpio)
    
    # Initialize Edge-AI engine
    if not os.path.exists(DEFAULT_MODEL_PATH):
        train_and_save_model(DEFAULT_MODEL_PATH)
    st.session_state.ai_engine = GreenhouseEdgeAI(DEFAULT_MODEL_PATH)

    # Telemetry and experiment history buffers
    st.session_state.history_data: List[Dict[str, Any]] = []
    st.session_state.experiment_logs: List[Dict[str, Any]] = []
    st.session_state.step_counter = 0

    # Initial sensor state
    st.session_state.current_crop = "Tomato"
    st.session_state.temp_input = 23.5
    st.session_state.hum_input = 58.0
    st.session_state.moist_input = 62.0
    st.session_state.lux_input = 850.0
    st.session_state.co2_input = 450.0
    st.session_state.active_preset_name = "Normal Condition"
    st.session_state.hil_feedback = True
    st.session_state.sensor_noise = True
    st.session_state.initialized = True

sensor_array: VirtualSensorArray = st.session_state.sensor_array
ai_engine: GreenhouseEdgeAI = st.session_state.ai_engine
gpio = st.session_state.gpio


# ============================================================================
# SIDEBAR CONTROLS & EXPERIMENT PRESETS
# ============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/greenhouse.png",
        width=64,
    )
    st.markdown("### 🌿 Edge-AI Greenhouse Controller")
    st.caption("Raspberry Pi 4 / 3B+ Digital Twin & HIL Testbed")
    
    # Hardware Mode Indicator
    mode_text = "🟢 Digital Twin HAL (BCM 17, 27, 22)" if IS_HARDWARE_EMULATED else "⚡ Physical Raspberry Pi Connected"
    st.info(mode_text)

    st.markdown("---")
    st.markdown("#### 1. Crop Selection")
    crop_names = list(CROP_PROFILES.keys())
    selected_crop = st.selectbox(
        "Cultivated Crop Profile:",
        crop_names,
        index=crop_names.index(st.session_state.current_crop),
        key="crop_selector",
    )
    st.session_state.current_crop = selected_crop
    crop_info = CROP_PROFILES[selected_crop]
    st.caption(f"🔬 *{crop_info['scientific_name']}*")

    with st.expander("Agronomic Threshold Reference", expanded=False):
        st.markdown(
            f"""
            - **Temp Range:** {crop_info['temp_opt'][0]}°C – {crop_info['temp_opt'][1]}°C *(Max Safe: {crop_info['temp_max_safe']}°C)*
            - **Humidity:** {crop_info['hum_opt'][0]}% – {crop_info['hum_opt'][1]}% *(Max Safe: {crop_info['hum_max_safe']}%)*
            - **Soil Moisture:** {crop_info['moist_opt'][0]}% – {crop_info['moist_opt'][1]}% *(Min Safe: {crop_info['moist_min_safe']}%)*
            - **Photoperiod Light:** {crop_info['light_opt'][0]} – {crop_info['light_opt'][1]} Lux *(Min: {crop_info['light_min_safe']} Lux)*
            """
        )

    st.markdown("---")
    st.markdown("#### 2. Section III Experiment Presets")
    st.caption("Inject specific stress conditions from paper:")

    def apply_preset(name: str):
        cfg = EXPERIMENTAL_PRESETS[name]
        st.session_state.temp_input = float(cfg["temperature"])
        st.session_state.hum_input = float(cfg["humidity"])
        st.session_state.moist_input = float(cfg["soil_moisture"])
        st.session_state.lux_input = float(cfg["light_lux"])
        st.session_state.co2_input = float(cfg["co2_ppm"])
        st.session_state.active_preset_name = name
        sensor_array.set_preset(name)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌱 Normal", use_container_width=True):
            apply_preset("Normal Condition")
    with c2:
        if st.button("🏜️ Dry Soil", use_container_width=True):
            apply_preset("Dry Soil Condition (< 35%)")

    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔥 High Temp", use_container_width=True):
            apply_preset("High Temperature Condition (> 30°C / > 80% RH)")
    with c4:
        if st.button("🌑 Low Light", use_container_width=True):
            apply_preset("Low Light Condition (< 300 Lux)")

    if st.button("⚠️ Compound Multi-Stress", use_container_width=True):
        apply_preset("Compound Multi-Parameter Stress")

    st.markdown("---")
    st.markdown("#### 3. Manual Sensor Overrides")
    manual_temp = st.slider(
        "Temperature (°C) [DHT22]",
        min_value=10.0,
        max_value=45.0,
        value=float(st.session_state.temp_input),
        step=0.5,
    )
    manual_hum = st.slider(
        "Relative Humidity (%) [DHT22]",
        min_value=15.0,
        max_value=100.0,
        value=float(st.session_state.hum_input),
        step=1.0,
    )
    manual_moist = st.slider(
        "Soil Moisture (%) [Capacitive V1.2]",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.moist_input),
        step=1.0,
    )
    manual_lux = st.slider(
        "Ambient Light (Lux) [LDR]",
        min_value=0.0,
        max_value=2000.0,
        value=float(st.session_state.lux_input),
        step=20.0,
    )
    manual_co2 = st.slider(
        "CO2 (ppm) [NDIR]",
        min_value=300.0,
        max_value=1500.0,
        value=float(st.session_state.co2_input),
        step=25.0,
    )

    # Sync manual changes
    st.session_state.temp_input = manual_temp
    st.session_state.hum_input = manual_hum
    st.session_state.moist_input = manual_moist
    st.session_state.lux_input = manual_lux
    st.session_state.co2_input = manual_co2
    sensor_array.set_manual_readings(manual_temp, manual_hum, manual_moist, manual_lux, manual_co2)

    st.markdown("---")
    st.markdown("#### 4. Digital Twin Simulation Engine")
    hil_feedback = st.checkbox("Closed-Loop HIL Feedback", value=st.session_state.hil_feedback)
    sensor_noise = st.checkbox("Sensor Measurement Noise", value=st.session_state.sensor_noise)
    st.session_state.hil_feedback = hil_feedback
    st.session_state.sensor_noise = sensor_noise
    sensor_array.noise_enabled = sensor_noise

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        step_pressed = st.button("⏱️ Next Tick (+1s)", use_container_width=True)
    with col_btn2:
        reset_pressed = st.button("🔄 Reset System", use_container_width=True)

    if reset_pressed:
        st.session_state.history_data.clear()
        st.session_state.experiment_logs.clear()
        st.session_state.step_counter = 0
        apply_preset("Normal Condition")
        st.rerun()


# ============================================================================
# SIMULATION STEPPING & SENSOR TELEMETRY READ
# ============================================================================
if step_pressed and st.session_state.hil_feedback:
    sensor_array.step_simulation(delta_seconds=1.5)
    st.session_state.temp_input = sensor_array.temperature
    st.session_state.hum_input = sensor_array.humidity
    st.session_state.moist_input = sensor_array.soil_moisture
    st.session_state.lux_input = sensor_array.light_lux
    st.session_state.co2_input = sensor_array.co2_ppm

# Read live virtual sensor payload
telemetry = sensor_array.read_all()
cur_temp = telemetry["temperature"]
cur_hum = telemetry["humidity"]
cur_moist = telemetry["soil_moisture"]
cur_lux = telemetry["light_lux"]
cur_co2 = telemetry["co2_ppm"]


# ============================================================================
# EDGE-AI INFERENCE & DUAL-CONTROL ARBITER
# ============================================================================
ai_result = ai_engine.predict(
    crop_name=selected_crop,
    temperature=cur_temp,
    humidity=cur_hum,
    soil_moisture=cur_moist,
    light_lux=cur_lux,
)

control_decision = ai_engine.evaluate_hybrid_control(
    crop_name=selected_crop,
    temperature=cur_temp,
    humidity=cur_hum,
    soil_moisture=cur_moist,
    light_lux=cur_lux,
    ai_result=ai_result,
)

# Apply decisions to Virtual GPIO BCM Pins (Actuators)
pump_state = HIGH if control_decision["water_pump"]["active"] else LOW
fan_state = HIGH if control_decision["cooling_fan"]["active"] else LOW
light_state = HIGH if control_decision["grow_light"]["active"] else LOW

gpio.output(PIN_WATER_PUMP, pump_state)
gpio.output(PIN_COOLING_FAN, fan_state)
gpio.output(PIN_GROW_LIGHT, light_state)

# Record history for plotting
st.session_state.step_counter += 1
step_num = st.session_state.step_counter
timestamp_str = datetime.now().strftime("%H:%M:%S")

history_record = {
    "step": step_num,
    "timestamp": timestamp_str,
    "temperature": cur_temp,
    "humidity": cur_hum,
    "soil_moisture": cur_moist,
    "light_lux": cur_lux,
    "co2_ppm": cur_co2,
    "water_pump": 1 if pump_state == HIGH else 0,
    "cooling_fan": 1 if fan_state == HIGH else 0,
    "grow_light": 1 if light_state == HIGH else 0,
}
st.session_state.history_data.append(history_record)
if len(st.session_state.history_data) > 60:
    st.session_state.history_data.pop(0)

# Record experiment log
log_entry = {
    "Step": step_num,
    "Timestamp": timestamp_str,
    "Crop": selected_crop,
    "Condition": ai_result["condition"],
    "Confidence (%)": ai_result["confidence"],
    "Edge Latency (ms)": ai_result["edge_latency_ms"],
    "Temp (°C)": cur_temp,
    "Humidity (%)": cur_hum,
    "Soil Moist (%)": cur_moist,
    "Light (Lux)": cur_lux,
    "Pump [GPIO 17]": "ON" if pump_state == HIGH else "OFF",
    "Fan [GPIO 27]": "ON" if fan_state == HIGH else "OFF",
    "Light [GPIO 22]": "ON" if light_state == HIGH else "OFF",
    "Control Mode": control_decision["control_mode"],
}
st.session_state.experiment_logs.append(log_entry)


# ============================================================================
# MAIN DASHBOARD INTERFACE
# ============================================================================

# TOP BANNER
st.markdown(
    f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h2 style="margin: 0; color: #ffffff; font-weight: 800; letter-spacing: -0.5px;">
                    🌿 Edge-AI Smart Greenhouse Control Center
                </h2>
                <p style="margin: 4px 0 0 0; color: #93c5fd; font-size: 0.95rem;">
                    IEEE Access Research Implementation &bull; Raspberry Pi Digital Twin HIL Architecture
                </p>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #bfdbfe; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem;">
                    Active Crop: <b>{selected_crop}</b>
                </span>
                <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7; padding: 6px 14px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; margin-left: 8px;">
                    Tick: #{step_num} | {timestamp_str}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 1. LIVE SENSOR TELEMETRY METRICS
# ----------------------------------------------------------------------------
opt_temp_mid = (crop_info["temp_opt"][0] + crop_info["temp_opt"][1]) / 2.0
opt_hum_mid = (crop_info["hum_opt"][0] + crop_info["hum_opt"][1]) / 2.0
opt_moist_mid = (crop_info["moist_opt"][0] + crop_info["moist_opt"][1]) / 2.0
opt_lux_mid = (crop_info["light_opt"][0] + crop_info["light_opt"][1]) / 2.0

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric(
        label="🌡️ Temperature [DHT22]",
        value=f"{cur_temp:.1f} °C",
        delta=f"{cur_temp - opt_temp_mid:+.1f} °C vs opt",
        delta_color="inverse" if abs(cur_temp - opt_temp_mid) > 4.0 else "normal",
    )
with col_m2:
    st.metric(
        label="💧 Humidity [DHT22]",
        value=f"{cur_hum:.1f} %",
        delta=f"{cur_hum - opt_hum_mid:+.1f} % vs opt",
        delta_color="inverse" if abs(cur_hum - opt_hum_mid) > 15.0 else "normal",
    )
with col_m3:
    st.metric(
        label="🌱 Soil Moisture [Capacitive]",
        value=f"{cur_moist:.1f} %",
        delta=f"{cur_moist - opt_moist_mid:+.1f} % vs opt",
        delta_color="inverse" if cur_moist < crop_info["moist_min_safe"] else "normal",
    )
with col_m4:
    st.metric(
        label="☀️ Light Level [LDR]",
        value=f"{cur_lux:.0f} Lux",
        delta=f"{cur_lux - opt_lux_mid:+.0f} Lux vs opt",
        delta_color="inverse" if cur_lux < crop_info["light_min_safe"] else "normal",
    )
with col_m5:
    st.metric(
        label="🫧 CO2 Concentration [NDIR]",
        value=f"{cur_co2:.0f} ppm",
        delta="Optimal" if cur_co2 < 800 else "Elevated",
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 2. EDGE-AI DIAGNOSTICS & HARDWARE ACTUATOR STATUS
# ----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("### 🧠 Edge-AI Diagnostics & Latency Benchmark")
    
    cond_name = ai_result["condition"]
    if cond_name == "Optimal":
        badge_class = "badge-optimal"
        badge_icon = "✅"
    elif "Compound" in cond_name:
        badge_class = "badge-danger"
        badge_icon = "🚨"
    else:
        badge_class = "badge-warning"
        badge_icon = "⚠️"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #0f172a; border: 1px solid #1e293b; padding: 16px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div>
                <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Predicted Microclimate State</span>
                <div style="margin-top: 6px;">
                    <span class="{badge_class}">{badge_icon} {cond_name}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Model Confidence</span>
                <div style="font-size: 2rem; font-weight: 800; color: #38bdf8; margin-top: 4px;">
                    {ai_result['confidence']:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Edge vs Cloud Latency Benchmark
    edge_ms = ai_result["edge_latency_ms"]
    cloud_ms = ai_result["cloud_latency_ms"]
    speedup = round(cloud_ms / max(0.1, edge_ms), 1)

    col_lat1, col_lat2, col_lat3 = st.columns(3)
    with col_lat1:
        st.markdown(
            f"""
            <div class="benchmark-card">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">⚡ Local Edge Latency</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #10b981; margin-top: 4px;">
                    {edge_ms:.2f} ms
                </div>
                <span style="color: #6ee7b7; font-size: 0.75rem;">ARM Cortex-A Core</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_lat2:
        st.markdown(
            f"""
            <div class="benchmark-card">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">☁️ Cloud Roundtrip Latency</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #f59e0b; margin-top: 4px;">
                    {cloud_ms:.1f} ms
                </div>
                <span style="color: #fcd34d; font-size: 0.75rem;">REST API + 4G/Wi-Fi</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_lat3:
        st.markdown(
            f"""
            <div class="benchmark-card">
                <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">🚀 Edge Speed Advantage</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8; margin-top: 4px;">
                    {speedup}x Faster
                </div>
                <span style="color: #7dd3fc; font-size: 0.75rem;">100% Offline Resilience</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Class Probability Distribution Chart
    prob_df = pd.DataFrame(
        [
            {"Condition": k, "Probability (%)": v}
            for k, v in ai_result["probabilities"].items()
        ]
    )
    
    fig_prob = go.Figure(
        go.Bar(
            x=prob_df["Probability (%)"],
            y=prob_df["Condition"],
            orientation="h",
            marker=dict(
                color=[
                    "#10b981" if c == "Optimal" else ("#ef4444" if "Compound" in c else "#f59e0b")
                    for c in prob_df["Condition"]
                ],
                line=dict(color="#334155", width=1),
            ),
            text=[f"{p:.1f}%" for p in prob_df["Probability (%)"]],
            textposition="auto",
        )
    )
    fig_prob.update_layout(
        title="Edge-AI Condition Class Probabilities",
        title_font_size=13,
        title_font_color="#cbd5e1",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=30, b=10),
        height=180,
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="#1e293b", ticksuffix="%"),
        yaxis=dict(autorange="reversed", color="#e2e8f0"),
    )
    st.plotly_chart(fig_prob, use_container_width=True)


with col_right:
    st.markdown("### 🔌 Raspberry Pi Actuators (BCM Relays)")
    
    # Active mode explanation banner
    if control_decision["override_triggers"]:
        st.warning(
            f"**⚠️ Deterministic Safety Override Triggered!**\n\n"
            + " &bull; " + "\n &bull; ".join(control_decision["override_triggers"])
        )
    else:
        st.success("**✅ Edge-AI Autonomous Control Mode:** Relays managed by ML predictions.")

    # Actuator Card 1: Water Pump [GPIO 17]
    pump_on = pump_state == HIGH
    pump_badge = '<span class="status-pill-on">● HIGH (PUMP ON)</span>' if pump_on else '<span class="status-pill-off">○ LOW (PUMP OFF)</span>'
    st.markdown(
        f"""
        <div class="actuator-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #f8fafc;">💧 Water Pump Relay</h4>
                    <span style="font-family: monospace; font-size: 0.8rem; color: #38bdf8;">BCM Pin: GPIO 17 &bull; Soil Irrigation</span>
                </div>
                <div>{pump_badge}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                Trigger Source: <b style="color: #e2e8f0;">{control_decision['water_pump']['source']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actuator Card 2: Cooling Fan [GPIO 27]
    fan_on = fan_state == HIGH
    fan_badge = '<span class="status-pill-on">● HIGH (FAN ACTIVE)</span>' if fan_on else '<span class="status-pill-off">○ LOW (FAN OFF)</span>'
    st.markdown(
        f"""
        <div class="actuator-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #f8fafc;">🌀 Cooling & Ventilation Fan</h4>
                    <span style="font-family: monospace; font-size: 0.8rem; color: #38bdf8;">BCM Pin: GPIO 27 &bull; Thermal / RH Regulation</span>
                </div>
                <div>{fan_badge}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                Trigger Source: <b style="color: #e2e8f0;">{control_decision['cooling_fan']['source']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actuator Card 3: Grow Light [GPIO 22]
    light_on = light_state == HIGH
    light_badge = '<span class="status-pill-on">● HIGH (GROW LIGHT ON)</span>' if light_on else '<span class="status-pill-off">○ LOW (LIGHT OFF)</span>'
    st.markdown(
        f"""
        <div class="actuator-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #f8fafc;">💡 Supplemental Grow Light</h4>
                    <span style="font-family: monospace; font-size: 0.8rem; color: #38bdf8;">BCM Pin: GPIO 22 &bull; Photoperiod Optimization</span>
                </div>
                <div>{light_badge}</div>
            </div>
            <div style="margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
                Trigger Source: <b style="color: #e2e8f0;">{control_decision['grow_light']['source']}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ----------------------------------------------------------------------------
# 3. INTERACTIVE PLOTLY TIME-SERIES & DIGITAL WAVEFORM CHARTS
# ----------------------------------------------------------------------------
st.markdown("### 📈 Real-Time Telemetry & Actuator Digital Waveforms")

if st.session_state.history_data:
    df_hist = pd.DataFrame(st.session_state.history_data)
    
    fig_time = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "Microclimate Environmental Telemetry (DHT22, Soil Moisture, LDR)",
            "Raspberry Pi Relay Digital States [0=LOW/OFF, 1=HIGH/ON]",
        ),
    )

    # Telemetry Traces
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["temperature"],
            name="Temperature (°C)",
            line=dict(color="#f87171", width=2),
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["humidity"],
            name="Humidity (%)",
            line=dict(color="#60a5fa", width=2),
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["soil_moisture"],
            name="Soil Moisture (%)",
            line=dict(color="#34d399", width=2.5),
            mode="lines+markers",
        ),
        row=1,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["light_lux"] / 10.0,
            name="Light (Lux ÷ 10)",
            line=dict(color="#fbbf24", width=1.5, dash="dot"),
            mode="lines",
        ),
        row=1,
        col=1,
    )

    # Actuator Step Waveforms
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["water_pump"] * 0.9,
            name="Water Pump [GPIO 17]",
            line=dict(color="#06b6d4", width=2.5, shape="hv"),
        ),
        row=2,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["cooling_fan"] * 0.8 + 0.05,
            name="Cooling Fan [GPIO 27]",
            line=dict(color="#f43f5e", width=2.5, shape="hv"),
        ),
        row=2,
        col=1,
    )
    fig_time.add_trace(
        go.Scatter(
            x=df_hist["step"],
            y=df_hist["grow_light"] * 0.7 + 0.1,
            name="Grow Light [GPIO 22]",
            line=dict(color="#eab308", width=2.5, shape="hv"),
        ),
        row=2,
        col=1,
    )

    fig_time.update_layout(
        paper_bgcolor="#0b111e",
        plot_bgcolor="#0f172a",
        height=480,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1", size=10),
        ),
    )
    fig_time.update_xaxes(showgrid=True, gridcolor="#1e293b", title_text="Simulation Step (Tick)", row=2, col=1)
    fig_time.update_yaxes(showgrid=True, gridcolor="#1e293b", row=1, col=1)
    fig_time.update_yaxes(
        tickvals=[0, 1],
        ticktext=["LOW", "HIGH"],
        range=[-0.2, 1.2],
        showgrid=True,
        gridcolor="#1e293b",
        row=2,
        col=1,
    )

    st.plotly_chart(fig_time, use_container_width=True)

st.markdown("---")


# ----------------------------------------------------------------------------
# 4. EXPERIMENT RESULTS LOG & CSV EXPORT
# ----------------------------------------------------------------------------
st.markdown("### 📋 Section III Research Experiment Log & Export")
st.caption(
    "Tabular evaluation data logged across live test iterations for IEEE paper verification and project review."
)

if st.session_state.experiment_logs:
    df_logs = pd.DataFrame(st.session_state.experiment_logs)
    
    # Render table with styling
    st.dataframe(
        df_logs.tail(15),
        use_container_width=True,
        hide_index=True,
    )

    col_exp1, col_exp2 = st.columns([1, 4])
    with col_exp1:
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Log (CSV)",
            data=csv_data,
            file_name=f"greenhouse_edge_ai_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp2:
        if st.button("🗑️ Clear Experiment History"):
            st.session_state.experiment_logs.clear()
            st.session_state.history_data.clear()
            st.session_state.step_counter = 0
            st.rerun()

