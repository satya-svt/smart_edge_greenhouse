"""
edge_ai_engine.py
==================
Edge-AI Model Training, Evaluation, and Low-Latency Inference Engine
for Smart Greenhouse Environment Classification and Actuation Control.

Research Context:
Based on the IEEE Access paper "Edge-AI Based Smart Greenhouse Monitoring and
Automated Plant Growth Control Using Raspberry Pi".

Key Architectural Features:
1. Synthetic agricultural dataset generator simulating 2,000+ greenhouse microclimate samples
   across 5 crops: Tomato, Chilli, Coriander, Spinach, Lettuce.
2. Lightweight Scikit-Learn RandomForestClassifier optimized for low-latency inference on
   resource-constrained ARM Cortex-A edge processors (Raspberry Pi 3B+ / 4 / 5).
3. Dual-mode / Hybrid Decision Matrix combining probabilistic Edge-AI condition classification
   with fast, deterministic safety threshold overrides.
4. Latency profiling measuring sub-5ms local Edge-AI execution versus ~180ms cloud round-trip delay.
"""

from __future__ import annotations
import os
import time
import random
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



# CROP AGRONOMIC PROFILES & OPTIMAL THRESHOLDS
# ============================================================================
CROP_PROFILES: Dict[str, Dict[str, Any]] = {
    "Tomato": {
        "scientific_name": "Solanum lycopersicum",
        "temp_opt": (20.0, 27.0),
        "hum_opt": (55.0, 70.0),
        "moist_opt": (55.0, 75.0),
        "light_opt": (600.0, 1100.0),
        "co2_opt": (400.0, 800.0),
        "temp_max_safe": 30.0,
        "hum_max_safe": 80.0,
        "moist_min_safe": 35.0,
        "light_min_safe": 300.0,
    },
    "Chilli": {
        "scientific_name": "Capsicum annuum",
        "temp_opt": (22.0, 30.0),
        "hum_opt": (50.0, 65.0),
        "moist_opt": (45.0, 65.0),
        "light_opt": (700.0, 1200.0),
        "co2_opt": (400.0, 750.0),
        "temp_max_safe": 33.0,
        "hum_max_safe": 80.0,
        "moist_min_safe": 30.0,
        "light_min_safe": 350.0,
    },
    "Coriander": {
        "scientific_name": "Coriandrum sativum",
        "temp_opt": (16.0, 24.0),
        "hum_opt": (50.0, 68.0),
        "moist_opt": (50.0, 70.0),
        "light_opt": (450.0, 850.0),
        "co2_opt": (400.0, 700.0),
        "temp_max_safe": 28.0,
        "hum_max_safe": 78.0,
        "moist_min_safe": 35.0,
        "light_min_safe": 280.0,
    },
    "Spinach": {
        "scientific_name": "Spinacia oleracea",
        "temp_opt": (14.0, 22.0),
        "hum_opt": (50.0, 65.0),
        "moist_opt": (55.0, 75.0),
        "light_opt": (400.0, 800.0),
        "co2_opt": (400.0, 700.0),
        "temp_max_safe": 26.0,
        "hum_max_safe": 78.0,
        "moist_min_safe": 38.0,
        "light_min_safe": 260.0,
    },
    "Lettuce": {
        "scientific_name": "Lactuca sativa",
        "temp_opt": (15.0, 21.0),
        "hum_opt": (50.0, 65.0),
        "moist_opt": (60.0, 80.0),
        "light_opt": (350.0, 750.0),
        "co2_opt": (400.0, 700.0),
        "temp_max_safe": 25.0,
        "hum_max_safe": 75.0,
        "moist_min_safe": 40.0,
        "light_min_safe": 250.0,
    },
}

# The 5 primary microclimate classification classes from Section III
CONDITION_CLASSES: List[str] = [
    "Optimal",
    "Water Stress (Dry Soil)",
    "Heat/Humidity Stress",
    "Low Light Stress",
    "Compound Stress",
]

CROP_LIST = list(CROP_PROFILES.keys())
CROP_TO_ID = {name: idx for idx, name in enumerate(CROP_LIST)}
ID_TO_CROP = {idx: name for idx, name in enumerate(CROP_LIST)}

FEATURE_COLUMNS = ["crop_id", "temperature", "humidity", "soil_moisture", "light_lux"]
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "greenhouse_edge_model.pkl")


# ============================================================================
# DATASET GENERATION
# ============================================================================
def generate_greenhouse_dataset(num_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Synthesizes a realistic 2,500-sample agricultural dataset across 5 crops and 5 stress states.
    Incorporates physical sensor bounds, Gaussian noise, and cross-parameter correlations.
    """
    np.random.seed(random_state)
    random.seed(random_state)

    samples_per_crop = num_samples // len(CROP_LIST)
    samples_per_class = samples_per_crop // len(CONDITION_CLASSES)

    records = []

    for crop_name in CROP_LIST:
        crop_id = CROP_TO_ID[crop_name]
        prof = CROP_PROFILES[crop_name]

        # 1. OPTIMAL CONDITION
        for _ in range(samples_per_class):
            t = np.random.uniform(prof["temp_opt"][0], prof["temp_opt"][1]) + np.random.normal(0, 0.3)
            h = np.random.uniform(prof["hum_opt"][0], prof["hum_opt"][1]) + np.random.normal(0, 0.8)
            sm = np.random.uniform(prof["moist_opt"][0], prof["moist_opt"][1]) + np.random.normal(0, 0.9)
            lux = np.random.uniform(prof["light_opt"][0], prof["light_opt"][1]) + np.random.normal(0, 15.0)
            records.append({
                "crop_name": crop_name,
                "crop_id": crop_id,
                "temperature": round(float(np.clip(t, 10.0, 42.0)), 2),
                "humidity": round(float(np.clip(h, 25.0, 98.0)), 2),
                "soil_moisture": round(float(np.clip(sm, 5.0, 95.0)), 2),
                "light_lux": round(float(np.clip(lux, 20.0, 2000.0)), 1),
                "condition": "Optimal",
                "label": 0,
            })

        # 2. WATER STRESS (DRY SOIL)
        for _ in range(samples_per_class):
            t = np.random.uniform(prof["temp_opt"][0], prof["temp_opt"][1] + 2.0)
            h = np.random.uniform(prof["hum_opt"][0] - 8.0, prof["hum_opt"][1])
            # Soil moisture severely low: 8% to 33%
            sm = np.random.uniform(8.0, prof["moist_min_safe"] - 2.0)
            lux = np.random.uniform(prof["light_opt"][0], prof["light_opt"][1])
            records.append({
                "crop_name": crop_name,
                "crop_id": crop_id,
                "temperature": round(float(np.clip(t, 10.0, 42.0)), 2),
                "humidity": round(float(np.clip(h, 20.0, 95.0)), 2),
                "soil_moisture": round(float(np.clip(sm, 5.0, 34.5)), 2),
                "light_lux": round(float(np.clip(lux, 300.0, 1800.0)), 1),
                "condition": "Water Stress (Dry Soil)",
                "label": 1,
            })

        # 3. HEAT / HUMIDITY STRESS
        for _ in range(samples_per_class):
            # Thermal overload > 30°C or humidity > 80%
            is_heat = np.random.choice([True, False, True])
            if is_heat:
                t = np.random.uniform(prof["temp_max_safe"] + 1.0, 41.0)
                h = np.random.uniform(60.0, 92.0)
            else:
                t = np.random.uniform(25.0, 33.0)
                h = np.random.uniform(prof["hum_max_safe"] + 2.0, 96.0)
            sm = np.random.uniform(prof["moist_min_safe"] + 5.0, 80.0)
            lux = np.random.uniform(prof["light_opt"][0], 1400.0)
            records.append({
                "crop_name": crop_name,
                "crop_id": crop_id,
                "temperature": round(float(np.clip(t, 28.0, 44.0)), 2),
                "humidity": round(float(np.clip(h, 45.0, 99.0)), 2),
                "soil_moisture": round(float(np.clip(sm, 36.0, 90.0)), 2),
                "light_lux": round(float(np.clip(lux, 310.0, 1900.0)), 1),
                "condition": "Heat/Humidity Stress",
                "label": 2,
            })

        # 4. LOW LIGHT STRESS
        for _ in range(samples_per_class):
            t = np.random.uniform(prof["temp_opt"][0] - 2.0, prof["temp_opt"][1])
            h = np.random.uniform(prof["hum_opt"][0], prof["hum_opt"][1] + 5.0)
            sm = np.random.uniform(prof["moist_min_safe"] + 5.0, 75.0)
            # Low light < 300 Lux (cloudy, evening, shaded)
            lux = np.random.uniform(20.0, prof["light_min_safe"] - 15.0)
            records.append({
                "crop_name": crop_name,
                "crop_id": crop_id,
                "temperature": round(float(np.clip(t, 12.0, 30.0)), 2),
                "humidity": round(float(np.clip(h, 40.0, 80.0)), 2),
                "soil_moisture": round(float(np.clip(sm, 36.0, 85.0)), 2),
                "light_lux": round(float(np.clip(lux, 10.0, 285.0)), 1),
                "condition": "Low Light Stress",
                "label": 3,
            })

        # 5. COMPOUND MULTI-PARAMETER STRESS
        for _ in range(samples_per_class):
            # Simultaneous moisture deficit + extreme heat or darkness
            t = np.random.uniform(prof["temp_max_safe"] + 2.0, 42.0)
            h = np.random.choice([np.random.uniform(20.0, 35.0), np.random.uniform(82.0, 96.0)])
            sm = np.random.uniform(8.0, 28.0)  # Very dry
            lux = np.random.choice([np.random.uniform(30.0, 220.0), np.random.uniform(1100.0, 1800.0)])
            records.append({
                "crop_name": crop_name,
                "crop_id": crop_id,
                "temperature": round(float(np.clip(t, 31.0, 44.0)), 2),
                "humidity": round(float(np.clip(h, 20.0, 99.0)), 2),
                "soil_moisture": round(float(np.clip(sm, 5.0, 32.0)), 2),
                "light_lux": round(float(np.clip(lux, 20.0, 2000.0)), 1),
                "condition": "Compound Stress",
                "label": 4,
            })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================
def train_and_save_model(
    model_path: str = DEFAULT_MODEL_PATH,
    num_samples: int = 2500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Trains a lightweight Random Forest Edge-AI Classifier and saves it.
    Returns metrics dictionary with accuracy, classification report, and confusion matrix.
    """
    df = generate_greenhouse_dataset(num_samples=num_samples, random_state=random_state)
    
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Edge-optimized hyper-parameters: fast inference tree depth and ensemble size
    clf = RandomForestClassifier(
        n_estimators=70,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_duration_sec = time.perf_counter() - t0

    # Predictions & metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=CONDITION_CLASSES, output_dict=True
    )
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Benchmark average inference latency over 100 single-sample inferences
    latencies_us = []
    sample_df = X_test.iloc[0:1]
    for _ in range(100):
        t_start = time.perf_counter_ns()
        _ = clf.predict(sample_df)
        latencies_us.append((time.perf_counter_ns() - t_start) / 1000.0)
    avg_inference_ms = float(np.mean(latencies_us) / 1000.0)

    # Bundle artifacts
    model_bundle = {
        "model": clf,
        "feature_columns": FEATURE_COLUMNS,
        "condition_classes": CONDITION_CLASSES,
        "crop_profiles": CROP_PROFILES,
        "crop_to_id": CROP_TO_ID,
        "id_to_crop": ID_TO_CROP,
        "metrics": {
            "accuracy": float(acc),
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "train_duration_sec": round(train_duration_sec, 4),
            "avg_inference_latency_ms": round(avg_inference_ms, 3),
            "classification_report": report,
            "confusion_matrix": conf_matrix,
        },
    }

    joblib.dump(model_bundle, model_path)

    return model_bundle["metrics"]


# ============================================================================
# RUNTIME EDGE-AI INFERENCE ENGINE & DUAL-CONTROL SYSTEM
# ============================================================================
class GreenhouseEdgeAI:
    """
    Local runtime Edge-AI Inference Engine and Actuator Control Arbiter.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, auto_train: bool = True):
        self.model_path = model_path
        self.bundle: Optional[Dict[str, Any]] = None
        self.model: Optional[RandomForestClassifier] = None
        self._load_or_train(auto_train)

    def _load_or_train(self, auto_train: bool) -> None:
        """Loads cached model if present, or trains a fresh model."""
        if os.path.exists(self.model_path):
            try:
                self.bundle = joblib.load(self.model_path)
                self.model = self.bundle["model"]
                return
            except Exception:
                pass
        
        if auto_train:
            train_and_save_model(model_path=self.model_path)
            self.bundle = joblib.load(self.model_path)
            self.model = self.bundle["model"]

    def predict(
        self,
        crop_name: str,
        temperature: float,
        humidity: float,
        soil_moisture: float,
        light_lux: float,
    ) -> Dict[str, Any]:
        """
        Runs real-time Edge-AI inference on single greenhouse sensor telemetry vector.
        Calculates exact inference latency and simulates cloud transmission latency.
        """
        if self.model is None or self.bundle is None:
            self._load_or_train(auto_train=True)

        crop_id = CROP_TO_ID.get(crop_name, 0)
        feature_df = pd.DataFrame([{
            "crop_id": crop_id,
            "temperature": float(temperature),
            "humidity": float(humidity),
            "soil_moisture": float(soil_moisture),
            "light_lux": float(light_lux),
        }], columns=FEATURE_COLUMNS)

        # Microsecond precision edge latency timing
        t_start = time.perf_counter_ns()
        probs = self.model.predict_proba(feature_df)[0]
        pred_label = int(np.argmax(probs))
        edge_latency_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        # Simulated cloud round-trip delay (typical LTE/Wi-Fi to cloud REST API: 140ms - 240ms)
        cloud_latency_ms = round(165.0 + random.uniform(-25.0, 45.0), 1)

        condition_name = CONDITION_CLASSES[pred_label]
        confidence = float(probs[pred_label])

        # Recommended AI actions based on condition
        ai_recommendations = {
            "water_pump": pred_label in (1, 4),   # Water Stress or Compound Stress
            "cooling_fan": pred_label in (2, 4),  # Heat/Hum Stress or Compound Stress
            "grow_light": pred_label in (3, 4),   # Low Light or Compound Stress
        }

        return {
            "condition": condition_name,
            "label": pred_label,
            "confidence": round(confidence * 100.0, 1),
            "probabilities": {
                CONDITION_CLASSES[i]: round(float(prob) * 100.0, 1)
                for i, prob in enumerate(probs)
            },
            "edge_latency_ms": round(edge_latency_ms, 2),
            "cloud_latency_ms": cloud_latency_ms,
            "ai_recommendations": ai_recommendations,
        }

    def evaluate_hybrid_control(
        self,
        crop_name: str,
        temperature: float,
        humidity: float,
        soil_moisture: float,
        light_lux: float,
        ai_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hybrid Edge-AI + Deterministic Safety Threshold Controller.
        Combines predictive Edge-AI inference with hard real-time safety limits:
        - If Soil Moisture < safe limit -> Water Pump Forced ON
        - If Temp > safe limit or Humidity > safe limit -> Cooling Fan Forced ON
        - If Light Lux < safe limit -> Grow Light Forced ON
        """
        prof = CROP_PROFILES.get(crop_name, CROP_PROFILES["Tomato"])
        ai_recs = ai_result["ai_recommendations"]

        override_triggers = []

        # 1. Irrigation safety check
        pump_active = ai_recs["water_pump"]
        if soil_moisture < prof["moist_min_safe"]:
            pump_active = True
            override_triggers.append(f"Low Soil Moisture Safety Override ({soil_moisture}% < {prof['moist_min_safe']}%)")

        # 2. Ventilation safety check
        fan_active = ai_recs["cooling_fan"]
        if temperature > prof["temp_max_safe"]:
            fan_active = True
            override_triggers.append(f"High Temp Safety Override ({temperature}°C > {prof['temp_max_safe']}°C)")
        elif humidity > prof["hum_max_safe"]:
            fan_active = True
            override_triggers.append(f"High Humidity Safety Override ({humidity}% > {prof['hum_max_safe']}%)")

        # 3. Supplemental lighting safety check
        light_active = ai_recs["grow_light"]
        if light_lux < prof["light_min_safe"]:
            light_active = True
            override_triggers.append(f"Low Photoperiod Safety Override ({light_lux} Lux < {prof['light_min_safe']} Lux)")

        control_mode = "Deterministic Safety Override" if override_triggers else "Edge-AI Autonomous"

        return {
            "control_mode": control_mode,
            "override_triggers": override_triggers,
            "water_pump": {
                "active": pump_active,
                "state_str": "HIGH (ON)" if pump_active else "LOW (OFF)",
                "source": "Safety Override" if any("Soil Moisture" in t for t in override_triggers) else "Edge-AI",
            },
            "cooling_fan": {
                "active": fan_active,
                "state_str": "HIGH (ON)" if fan_active else "LOW (OFF)",
                "source": "Safety Override" if any("Temp" in t or "Humidity" in t for t in override_triggers) else "Edge-AI",
            },
            "grow_light": {
                "active": light_active,
                "state_str": "HIGH (ON)" if light_active else "LOW (OFF)",
                "source": "Safety Override" if any("Photoperiod" in t for t in override_triggers) else "Edge-AI",
            },
        }


# ============================================================================
# CLI ENTRYPOINT FOR TRAINING & VERIFICATION
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("   Edge-AI Greenhouse Model Training & Evaluation (IEEE Access)   ")
    print("=" * 70)
    print("Generating 2,500 synthetic microclimate samples across 5 crop profiles...")
    metrics = train_and_save_model()
    
    print("\n[+] Model Training Complete!")
    print(f"    - Training Dataset Size: {metrics['training_samples']} samples")
    print(f"    - Test Dataset Size:     {metrics['testing_samples']} samples")
    print(f"    - Overall Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"    - Training Duration:     {metrics['train_duration_sec']:.3f} s")
    print(f"    - Avg Inference Latency: {metrics['avg_inference_latency_ms']:.3f} ms (Edge)")
    print("\nClassification Report:")
    report = metrics["classification_report"]
    for cls in CONDITION_CLASSES:
        if cls in report:
            p = report[cls]["precision"] * 100
            r = report[cls]["recall"] * 100
            f1 = report[cls]["f1-score"] * 100
            print(f"    {cls:<26} | Precision: {p:5.1f}% | Recall: {r:5.1f}% | F1: {f1:5.1f}%")

    print("\nRunning Test Inference on Tomato under Dry Soil condition:")
    engine = GreenhouseEdgeAI()
    sample_pred = engine.predict("Tomato", temperature=24.5, humidity=50.0, soil_moisture=22.0, light_lux=750.0)
    decision = engine.evaluate_hybrid_control("Tomato", 24.5, 50.0, 22.0, 750.0, sample_pred)
    
    print(f"    - Predicted Condition:  {sample_pred['condition']} ({sample_pred['confidence']}% confidence)")
    print(f"    - Edge Latency:         {sample_pred['edge_latency_ms']} ms  vs  Cloud Latency: {sample_pred['cloud_latency_ms']} ms")
    print(f"    - Water Pump [GPIO 17]: {decision['water_pump']['state_str']}")
    print(f"    - Cooling Fan [GPIO 27]:{decision['cooling_fan']['state_str']}")
    print(f"    - Grow Light [GPIO 22]: {decision['grow_light']['state_str']}")
    print(f"    - Decision Mode:        {decision['control_mode']}")
    print("=" * 70)
