"""
test_app.py
Comprehensive test suite verifying the Mental Health & Stress Score Predictor
endpoints, models, calculations, and web server routes.
"""

import json
import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from app import app
from mental_health_engine import MentalHealthPredictor, MENTAL_HEALTH_PRESETS, QUESTIONS_METADATA

def run_tests():
    print("========================================")
    print("🧪 Running Mental Health Predictor Tests")
    print("========================================")
    
    # 1. Test engine directly
    print("\n1. Testing MentalHealthPredictor engine directly...")
    engine = MentalHealthPredictor()
    assert engine.model_data is not None, "Model data should be loaded"
    print("   ✓ ML Model loaded successfully")
    
    for p_key, p_val in MENTAL_HEALTH_PRESETS.items():
        res = engine.predict(p_val['data'])
        mh_score = res['mental_health_score']
        stress_score = res['stress_score']
        print(f"   ✓ Persona '{p_key}': MH Score = {mh_score}, Stress Score = {stress_score}, Tier = {res['mental_health_tier']['title']}")
        assert 0 <= mh_score <= 100, f"MH score {mh_score} out of bounds"
        assert 0 <= stress_score <= 100, f"Stress score {stress_score} out of bounds"
        assert len(res['domains']) == 6, "Expected 6 domains"
        assert len(res['radar_values']) == 6, "Expected 6 radar values"
        assert 'action_plan' in res, "Expected action plan"
        assert 'immediate_toolkit' in res['action_plan'], "Expected immediate toolkit"
        assert 'resources' in res['action_plan'], "Expected resources"
    
    # 2. Test Flask Client
    print("\n2. Testing Flask routes & API endpoints...")
    client = app.test_client()
    
    # GET /
    res = client.get('/')
    assert res.status_code == 200, f"GET / returned {res.status_code}"
    assert b"MindVitalis" in res.data, "Brand should be in index.html"
    assert b"Mental Health &amp; Stress" in res.data or b"Mental Health & Stress" in res.data
    print("   ✓ GET / returned 200 OK with MindVitalis content")
    
    # GET /mental-health
    res = client.get('/mental-health')
    assert res.status_code == 200, f"GET /mental-health returned {res.status_code}"
    print("   ✓ GET /mental-health returned 200 OK")
    
    # GET /api/mental-health/presets
    res = client.get('/api/mental-health/presets')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['status'] == 'success'
    assert 'thriving' in data['presets']
    assert 'burnout' in data['presets']
    print(f"   ✓ GET /api/mental-health/presets returned {len(data['presets'])} presets")
    
    # GET /api/mental-health/questions
    res = client.get('/api/mental-health/questions')
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['status'] == 'success'
    assert len(data['questions']) == 15
    print(f"   ✓ GET /api/mental-health/questions returned {len(data['questions'])} validated questions")
    
    # POST /api/mental-health/predict
    sample_payload = MENTAL_HEALTH_PRESETS['thriving']['data']
    res = client.post('/api/mental-health/predict', json=sample_payload)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data['status'] == 'success'
    pred = data['data']
    assert pred['mental_health_score'] >= 75
    assert pred['stress_score'] <= 35
    print(f"   ✓ POST /api/mental-health/predict returned MH: {pred['mental_health_score']}, Stress: {pred['stress_score']}")
    
    # Test high stress persona
    sample_burnout = MENTAL_HEALTH_PRESETS['burnout']['data']
    res_b = client.post('/api/mental-health/predict', json=sample_burnout)
    assert res_b.status_code == 200
    pred_b = json.loads(res_b.data)['data']
    assert pred_b['stress_score'] > 50, "Burnout persona should have elevated stress"
    print(f"   ✓ POST /api/mental-health/predict for Burnout returned Stress: {pred_b['stress_score']}, MH: {pred_b['mental_health_score']}")

    # 3. Test Physical Health Prediction is still intact
    print("\n3. Testing Physical Health prediction backwards compatibility...")
    physical_payload = {
        "age": 30,
        "gender": "male",
        "height": 175,
        "weight": 70,
        "systolic_bp": 118,
        "diastolic_bp": 76,
        "resting_hr": 65,
        "glucose": 88,
        "cholesterol": 180,
        "activity_mins": 180,
        "daily_steps": 8500,
        "sleep_hours": 7.5,
        "sleep_quality": 4,
        "water_liters": 2.5,
        "diet_quality": 4,
        "smoking_status": "never",
        "alcohol_intake": "none",
        "stress_level": 3,
        "screen_time": 5.5,
        "family_cvd": False,
        "family_diabetes": False,
        "family_hypertension": False
    }
    res_p = client.post('/api/predict', json=physical_payload)
    assert res_p.status_code == 200
    data_p = json.loads(res_p.data)
    assert data_p['status'] == 'success'
    print(f"   ✓ Physical Health Predictor intact: Overall Score = {data_p['data']['overall_score']}")

    print("\n========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("========================================")

if __name__ == '__main__':
    run_tests()
