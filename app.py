"""
app.py
Production Flask server for Health Score Predictor & Future Action Planner.
Provides REST API endpoints for score inference, preset persona profiles, and renders
the responsive modern UI.
"""

from flask import Flask, render_template, request, jsonify
from ml_engine import HealthScorePredictor
import os

app = Flask(__name__)
predictor = HealthScorePredictor()

# Preset demo personas for fast 1-click exploration by the user
PRESET_PERSONAS = {
    "athlete": {
        "title": "🏃 Athletic / Optimal Vitality",
        "description": "Active runner with excellent vitals, high hydration, and restorative sleep.",
        "data": {
            "age": 28,
            "gender": "male",
            "height": 178,
            "weight": 72,
            "systolic_bp": 114,
            "diastolic_bp": 74,
            "resting_hr": 54,
            "glucose": 84,
            "cholesterol": 168,
            "activity_mins": 320,
            "daily_steps": 12500,
            "sleep_hours": 8.0,
            "sleep_quality": 5,
            "water_liters": 3.2,
            "diet_quality": 5,
            "smoking_status": "never",
            "alcohol_intake": "light",
            "stress_level": 3,
            "screen_time": 4.5,
            "family_cvd": False,
            "family_diabetes": False,
            "family_hypertension": False
        }
    },
    "desk_worker": {
        "title": "💼 Sedentary Tech Professional",
        "description": "Desk job, elevated screen time, moderate stress, and irregular sleep schedule.",
        "data": {
            "age": 36,
            "gender": "male",
            "height": 175,
            "weight": 83,
            "systolic_bp": 128,
            "diastolic_bp": 84,
            "resting_hr": 78,
            "glucose": 104,
            "cholesterol": 208,
            "activity_mins": 60,
            "daily_steps": 4200,
            "sleep_hours": 6.2,
            "sleep_quality": 3,
            "water_liters": 1.4,
            "diet_quality": 2,
            "smoking_status": "never",
            "alcohol_intake": "moderate",
            "stress_level": 7,
            "screen_time": 10.5,
            "family_cvd": False,
            "family_diabetes": True,
            "family_hypertension": False
        }
    },
    "metabolic_risk": {
        "title": "⚠️ Elevated Metabolic & Cardio Risk",
        "description": "High blood pressure, elevated fasting glucose, high BMI, and poor sleep.",
        "data": {
            "age": 52,
            "gender": "female",
            "height": 162,
            "weight": 88,
            "systolic_bp": 146,
            "diastolic_bp": 94,
            "resting_hr": 84,
            "glucose": 138,
            "cholesterol": 242,
            "activity_mins": 30,
            "daily_steps": 2800,
            "sleep_hours": 5.5,
            "sleep_quality": 2,
            "water_liters": 1.0,
            "diet_quality": 2,
            "smoking_status": "occasional",
            "alcohol_intake": "moderate",
            "stress_level": 8,
            "screen_time": 7.0,
            "family_cvd": True,
            "family_diabetes": True,
            "family_hypertension": True
        }
    },
    "senior_wellness": {
        "title": "🧘 Senior Proactive Wellness",
        "description": "Mindful senior walking daily, clean Mediterranean diet, and moderate BP management.",
        "data": {
            "age": 64,
            "gender": "female",
            "height": 160,
            "weight": 61,
            "systolic_bp": 124,
            "diastolic_bp": 78,
            "resting_hr": 66,
            "glucose": 94,
            "cholesterol": 192,
            "activity_mins": 190,
            "daily_steps": 8200,
            "sleep_hours": 7.3,
            "sleep_quality": 4,
            "water_liters": 2.2,
            "diet_quality": 4,
            "smoking_status": "former",
            "alcohol_intake": "none",
            "stress_level": 3,
            "screen_time": 3.0,
            "family_cvd": False,
            "family_diabetes": False,
            "family_hypertension": True
        }
    }
}

@app.route('/')
def index():
    return render_template('index.html', presets=PRESET_PERSONAS)

@app.route('/api/presets', methods=['GET'])
def get_presets():
    return jsonify({
        'status': 'success',
        'presets': PRESET_PERSONAS
    })

@app.route('/api/predict', methods=['POST'])
def predict_health():
    try:
        user_input = request.get_json(force=True)
        if not user_input:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        result = predictor.predict(user_input)
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        app.logger.error(f"Error during health prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Health Score Predictor on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
