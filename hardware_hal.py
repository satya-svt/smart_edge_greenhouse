"""
hardware_hal.py
================
Hardware Abstraction Layer (HAL) & Digital Twin Simulation for Raspberry Pi Smart Greenhouse.

This module provides:
1. `VirtualRPiGPIO`: A drop-in digital twin emulator for `RPi.GPIO` mimicking BCM pinouts,
   mode configuration, pin directions, digital reads/writes, and state tracking.
   BCM Pin mapping:
     - GPIO 17: Water Pump Relay (Irrigation)
     - GPIO 27: Cooling Fan Relay (Ventilation / Temperature-Humidity regulation)
     - GPIO 22: Artificial Grow Light Relay (Photoperiod Supplement)
2. `VirtualSensorArray`: Simulates DHT22 (Temp & Humidity), Capacitive Soil Moisture,
   LDR (Lux), and CO2 (ppm) sensors with experimental condition presets from Section III
   (Visual Abstract) of the IEEE Access research paper and closed-loop actuator feedback.
3. Seamless hardware fallback: Auto-detects physical Raspberry Pi hardware with RPi.GPIO
   and falls back cleanly to the Digital Twin HAL.
"""

from __future__ import annotations
import math
import random
import time
from typing import Dict, Any, Optional, Tuple, List, Callable


# ============================================================================
# RPi.GPIO CONSTANTS & ENUMS
# ============================================================================
BCM = 11
BOARD = 10

OUT = 0
IN = 1

HIGH = 1
LOW = 0

PUD_OFF = 20
PUD_DOWN = 21
PUD_UP = 22

RISING = 31
FALLING = 32
BOTH = 33

# Hardware BCM Pin Allocations matching the research paper
PIN_WATER_PUMP = 17   # Water pump relay (Active HIGH / LOW configurable)
PIN_COOLING_FAN = 27  # Ventilation cooling fan relay
PIN_GROW_LIGHT = 22   # Artificial LED grow light relay

PIN_NAMES = {
    PIN_WATER_PUMP: "Water Pump Relay",
    PIN_COOLING_FAN: "Cooling Fan Relay",
    PIN_GROW_LIGHT: "Grow Light Relay",
}


# ============================================================================
# VIRTUAL RASPBERRY PI GPIO (DIGITAL TWIN)
# ============================================================================
class VirtualRPiGPIO:
    """
    Virtual drop-in simulator for RPi.GPIO.
    Provides identical method signatures and internal state tracking for Digital Twin execution.
    """
    BCM = BCM
    BOARD = BOARD
    OUT = OUT
    IN = IN
    HIGH = HIGH
    LOW = LOW
    PUD_OFF = PUD_OFF
    PUD_DOWN = PUD_DOWN
    PUD_UP = PUD_UP
    RISING = RISING
    FALLING = FALLING
    BOTH = BOTH

    def __init__(self):
        self._mode: Optional[int] = None
        self._pin_directions: Dict[int, int] = {}
        self._pin_states: Dict[int, int] = {}
        self._pin_pulls: Dict[int, int] = {}
        self._warnings: bool = False
        self._event_callbacks: Dict[int, List[Callable]] = {}
        self._state_history: List[Dict[str, Any]] = []

    def setmode(self, mode: int) -> None:
        """Sets pin numbering mode: GPIO.BCM or GPIO.BOARD."""
        if mode not in (BCM, BOARD):
            raise ValueError(f"Invalid mode {mode}. Expected GPIO.BCM or GPIO.BOARD.")
        self._mode = mode

    def getmode(self) -> Optional[int]:
        """Returns the current pin numbering mode."""
        return self._mode

    def setwarnings(self, state: bool) -> None:
        """Enables or disables GPIO warnings."""
        self._warnings = bool(state)

    def setup(self, channel: int, direction: int, pull_up_down: int = PUD_OFF, initial: Optional[int] = None) -> None:
        """
        Configures a GPIO channel as IN or OUT.
        """
        if direction not in (IN, OUT):
            raise ValueError(f"Invalid direction {direction}. Must be GPIO.IN or GPIO.OUT.")
        self._pin_directions[channel] = direction
        self._pin_pulls[channel] = pull_up_down
        
        # Set initial value for output or pull state for input
        if direction == OUT:
            state = LOW if initial is None else initial
            self._pin_states[channel] = state
        else:
            if pull_up_down == PUD_UP:
                self._pin_states[channel] = HIGH
            elif pull_up_down == PUD_DOWN:
                self._pin_states[channel] = LOW
            else:
                self._pin_states[channel] = LOW

    def output(self, channel: int, state: int) -> None:
        """
        Sets the digital output state (HIGH or LOW) of a configured output pin.
        """
        if channel not in self._pin_directions:
            raise RuntimeError(f"The GPIO channel {channel} has not been set up as an OUTPUT.")
        if self._pin_directions[channel] != OUT:
            raise RuntimeError(f"The GPIO channel {channel} is configured as INPUT, cannot write output.")
        
        state_val = HIGH if state else LOW
        prev_state = self._pin_states.get(channel, None)
        self._pin_states[channel] = state_val
        
        # Log state transition for Digital Twin telemetry
        self._state_history.append({
            "timestamp": time.time(),
            "pin": channel,
            "pin_name": PIN_NAMES.get(channel, f"GPIO_{channel}"),
            "previous": prev_state,
            "current": state_val,
        })
        # Keep history capped at 1000 items
        if len(self._state_history) > 1000:
            self._state_history.pop(0)

    def input(self, channel: int) -> int:
        """
        Reads the digital state (HIGH or LOW) of a pin.
        """
        return self._pin_states.get(channel, LOW)

    def cleanup(self, channel: Optional[int] = None) -> None:
        """
        Resets GPIO configuration and states.
        """
        if channel is None:
            self._pin_directions.clear()
            self._pin_states.clear()
            self._pin_pulls.clear()
            self._event_callbacks.clear()
            self._mode = None
        else:
            self._pin_directions.pop(channel, None)
            self._pin_states.pop(channel, None)
            self._pin_pulls.pop(channel, None)
            self._event_callbacks.pop(channel, None)

    # ------------------------------------------------------------------------
    # Digital Twin Inspection Methods
    # ------------------------------------------------------------------------
    def get_pin_state(self, channel: int) -> int:
        """Convenience method to retrieve the current state of a pin."""
        return self._pin_states.get(channel, LOW)

    def get_all_actuator_states(self) -> Dict[str, Any]:
        """Returns snapshot of the 3 primary greenhouse relays."""
        return {
            "water_pump": {
                "pin": PIN_WATER_PUMP,
                "name": PIN_NAMES[PIN_WATER_PUMP],
                "state": self._pin_states.get(PIN_WATER_PUMP, LOW),
                "is_active": self._pin_states.get(PIN_WATER_PUMP, LOW) == HIGH,
            },
            "cooling_fan": {
                "pin": PIN_COOLING_FAN,
                "name": PIN_NAMES[PIN_COOLING_FAN],
                "state": self._pin_states.get(PIN_COOLING_FAN, LOW),
                "is_active": self._pin_states.get(PIN_COOLING_FAN, LOW) == HIGH,
            },
            "grow_light": {
                "pin": PIN_GROW_LIGHT,
                "name": PIN_NAMES[PIN_GROW_LIGHT],
                "state": self._pin_states.get(PIN_GROW_LIGHT, LOW),
                "is_active": self._pin_states.get(PIN_GROW_LIGHT, LOW) == HIGH,
            },
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the recent GPIO state transition history."""
        return self._state_history[-limit:]


# Instantiate singleton virtual GPIO
virtual_gpio = VirtualRPiGPIO()

# ----------------------------------------------------------------------------
# Seamless Hardware Detection & Export
# ----------------------------------------------------------------------------
try:
    import RPi.GPIO as _RealRPiGPIO  # type: ignore
    GPIO = _RealRPiGPIO
    IS_HARDWARE_EMULATED = False
except (ImportError, RuntimeError):
    GPIO = virtual_gpio
    IS_HARDWARE_EMULATED = True


def initialize_hardware(gpio_driver=None) -> Any:
    """
    Initializes Raspberry Pi GPIO pin layout for the greenhouse actuators.
    Defaults to the active GPIO driver (virtual or real).
    """
    driver = gpio_driver or GPIO
    driver.setwarnings(False)
    driver.setmode(BCM)
    driver.setup(PIN_WATER_PUMP, OUT, initial=LOW)
    driver.setup(PIN_COOLING_FAN, OUT, initial=LOW)
    driver.setup(PIN_GROW_LIGHT, OUT, initial=LOW)
    return driver


# ============================================================================
# EXPERIMENTAL CONDITION PRESETS (Section III Visual Abstract)
# ============================================================================
EXPERIMENTAL_PRESETS = {
    "Normal Condition": {
        "description": "Balanced optimal climate for greenhouse cultivation.",
        "temperature": 23.5,
        "humidity": 58.0,
        "soil_moisture": 62.0,
        "light_lux": 850.0,
        "co2_ppm": 450.0,
    },
    "Dry Soil Condition (< 35%)": {
        "description": "Severe root zone moisture deficit triggering irrigation pump.",
        "temperature": 24.5,
        "humidity": 52.0,
        "soil_moisture": 22.0,
        "light_lux": 800.0,
        "co2_ppm": 430.0,
    },
    "High Temperature Condition (> 30°C / > 80% RH)": {
        "description": "Thermal and relative humidity overload triggering cooling fan.",
        "temperature": 34.8,
        "humidity": 86.5,
        "soil_moisture": 54.0,
        "light_lux": 920.0,
        "co2_ppm": 490.0,
    },
    "Low Light Condition (< 300 Lux)": {
        "description": "Insufficient solar irradiance triggering artificial grow lights.",
        "temperature": 21.8,
        "humidity": 61.0,
        "soil_moisture": 59.0,
        "light_lux": 160.0,
        "co2_ppm": 420.0,
    },
    "Compound Multi-Parameter Stress": {
        "description": "Simultaneous severe moisture deficit, heatwave, and darkness.",
        "temperature": 35.5,
        "humidity": 83.0,
        "soil_moisture": 19.5,
        "light_lux": 180.0,
        "co2_ppm": 560.0,
    },
}


# ============================================================================
# VIRTUAL SENSOR ARRAY (DIGITAL TWIN WITH CLOSED-LOOP DYNAMICS)
# ============================================================================
class VirtualSensorArray:
    """
    Simulates physical sensor telemetry from:
    - DHT22: Temperature (°C), Humidity (%)
    - Capacitive Soil Moisture Sensor: V1.2 (%)
    - LDR Sensor: Ambient Light (Lux)
    - MQ-135 / NDIR CO2 Sensor: Carbon Dioxide (ppm)

    Features:
    - Realistic Gaussian measurement noise
    - Closed-loop physical dynamics responding to actuator relay states
    - Direct injection of Section III experimental stress conditions
    """

    def __init__(self, gpio_driver: Optional[VirtualRPiGPIO] = None):
        self.gpio = gpio_driver or virtual_gpio
        # State variables
        self.temperature = 23.5
        self.humidity = 58.0
        self.soil_moisture = 60.0
        self.light_lux = 850.0
        self.co2_ppm = 450.0

        # Physical dynamics coefficients
        self.ambient_temp = 24.0
        self.ambient_lux = 750.0
        self.noise_enabled = True
        self.last_update_time = time.time()

    def set_preset(self, preset_name: str) -> bool:
        """Injects one of the 5 Section III experimental conditions."""
        if preset_name in EXPERIMENTAL_PRESETS:
            cfg = EXPERIMENTAL_PRESETS[preset_name]
            self.temperature = float(cfg["temperature"])
            self.humidity = float(cfg["humidity"])
            self.soil_moisture = float(cfg["soil_moisture"])
            self.light_lux = float(cfg["light_lux"])
            self.co2_ppm = float(cfg["co2_ppm"])
            return True
        return False

    def set_manual_readings(
        self,
        temperature: float,
        humidity: float,
        soil_moisture: float,
        light_lux: float,
        co2_ppm: float = 450.0,
    ) -> None:
        """Allows direct manual overriding of environmental parameters."""
        self.temperature = float(temperature)
        self.humidity = float(humidity)
        self.soil_moisture = float(soil_moisture)
        self.light_lux = float(light_lux)
        self.co2_ppm = float(co2_ppm)

    def step_simulation(self, delta_seconds: float = 1.0) -> None:
        """
        Calculates physical feedback dynamics based on active relay states:
        - Water Pump (GPIO 17 ON) -> Soil moisture increases; OFF -> Slow dry-out
        - Cooling Fan (GPIO 27 ON) -> Temp drops towards 22°C, Humidity vents
        - Grow Light (GPIO 22 ON) -> Lux increases by +500 Lux
        """
        pump_on = self.gpio.input(PIN_WATER_PUMP) == HIGH
        fan_on = self.gpio.input(PIN_COOLING_FAN) == HIGH
        light_on = self.gpio.input(PIN_GROW_LIGHT) == HIGH

        dt = max(0.1, min(delta_seconds, 5.0))

        # 1. Soil moisture dynamics
        if pump_on:
            # Active irrigation replenishes root zone
            self.soil_moisture = min(92.0, self.soil_moisture + 3.0 * dt)
        else:
            # Natural evapotranspiration loss proportional to temperature
            loss_rate = 0.05 * (1.0 + max(0.0, self.temperature - 25.0) * 0.1)
            self.soil_moisture = max(5.0, self.soil_moisture - loss_rate * dt)

        # 2. Temperature & Humidity ventilation dynamics
        if fan_on:
            # Cooling fan circulates fresh air, dropping temp towards 22°C
            self.temperature = max(21.0, self.temperature - 0.75 * dt)
            self.humidity = max(45.0, self.humidity - 1.2 * dt)
        else:
            # Greenhouse heat accumulation
            if self.temperature < self.ambient_temp:
                self.temperature = min(self.ambient_temp, self.temperature + 0.1 * dt)

        # 3. Light dynamics
        target_light = self.ambient_lux + (550.0 if light_on else 0.0)
        # Smooth transition to target lux
        self.light_lux = self.light_lux + (target_light - self.light_lux) * min(1.0, 0.4 * dt)

        # 4. CO2 dynamics (slight natural respiration/ventilation drift)
        if fan_on:
            self.co2_ppm = max(400.0, self.co2_ppm - 5.0 * dt)
        else:
            self.co2_ppm = min(1200.0, self.co2_ppm + 1.0 * dt)

    def read_dht22(self) -> Tuple[float, float]:
        """
        Simulates reading digital Temperature and Humidity from DHT22.
        Includes typical sensor measurement noise (±0.2°C, ±0.5% RH).
        """
        temp = self.temperature
        hum = self.humidity
        if self.noise_enabled:
            temp += random.gauss(0.0, 0.15)
            hum += random.gauss(0.0, 0.4)
        return round(float(temp), 2), round(float(max(10.0, min(100.0, hum))), 2)

    def read_soil_moisture(self) -> float:
        """
        Simulates reading Capacitive Soil Moisture Sensor (0.0 to 100.0%).
        """
        sm = self.soil_moisture
        if self.noise_enabled:
            sm += random.gauss(0.0, 0.25)
        return round(float(max(0.0, min(100.0, sm))), 2)

    def read_light_lux(self) -> float:
        """
        Simulates reading LDR calibrated Lux level.
        """
        lux = self.light_lux
        if self.noise_enabled:
            lux += random.gauss(0.0, 5.0)
        return round(float(max(0.0, lux)), 1)

    def read_co2(self) -> float:
        """
        Simulates reading NDIR/MQ CO2 ppm concentration.
        """
        co2 = self.co2_ppm
        if self.noise_enabled:
            co2 += random.gauss(0.0, 3.0)
        return round(float(max(300.0, co2)), 1)

    def read_all(self) -> Dict[str, float]:
        """Returns clean telemetry payload of all greenhouse sensors."""
        t, h = self.read_dht22()
        sm = self.read_soil_moisture()
        lux = self.read_light_lux()
        co2 = self.read_co2()
        return {
            "temperature": t,
            "humidity": h,
            "soil_moisture": sm,
            "light_lux": lux,
            "co2_ppm": co2,
        }

