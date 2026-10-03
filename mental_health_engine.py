"""
mental_health_engine.py
Predicts Mental Health Score (0-100) and Stress Score (0-100) using a trained
Machine Learning RandomForest ensemble combined with evidence-based psychometric frameworks
(PSS-10, GAD-7, PHQ-9, WHO-5, Maslach Burnout dimensions).
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'mental_health_model.joblib')

# Preset Personas for 1-Click Evaluation
MENTAL_HEALTH_PRESETS = {
    "thriving": {
        "title": "🌱 Thriving & Mindful",
        "description": "Balanced emotional state, high resilience, restful sleep, and positive social anchors.",
        "data": {
            "q_anhedonia": 0,
            "q_depressed": 0,
            "q_optimism": 4,
            "q_overwhelmed": 1,
            "q_uncontrollable": 0,
            "q_coping_confidence": 4,
            "q_anxious": 1,
            "q_worry": 0,
            "q_somatic": 0,
            "q_sleep_issues": 0,
            "q_fatigue": 1,
            "q_burnout": 0,
            "q_concentration": 0,
            "q_isolation": 0,
            "q_coping_habits": 4,
            "sleep_hours": 8.0,
            "work_screen_hours": 5.5
        }
    },
    "burnout": {
        "title": "💼 Career Burnout & Fatigue",
        "description": "Tech/corporate professional with 11h screen time, cognitive drain, brain fog, and chronic fatigue.",
        "data": {
            "q_anhedonia": 2,
            "q_depressed": 2,
            "q_optimism": 1,
            "q_overwhelmed": 3,
            "q_uncontrollable": 3,
            "q_coping_confidence": 2,
            "q_anxious": 2,
            "q_worry": 2,
            "q_somatic": 2,
            "q_sleep_issues": 3,
            "q_fatigue": 4,
            "q_burnout": 4,
            "q_concentration": 3,
            "q_isolation": 2,
            "q_coping_habits": 1,
            "sleep_hours": 5.8,
            "work_screen_hours": 11.5
        }
    },
    "high_anxiety": {
        "title": "😰 Acute Anxiety & Insomnia",
        "description": "High nervous arousal, racing thoughts, muscle tension, and severe sleep disturbances.",
        "data": {
            "q_anhedonia": 2,
            "q_depressed": 2,
            "q_optimism": 1,
            "q_overwhelmed": 4,
            "q_uncontrollable": 3,
            "q_coping_confidence": 1,
            "q_anxious": 4,
            "q_worry": 4,
            "q_somatic": 4,
            "q_sleep_issues": 4,
            "q_fatigue": 3,
            "q_burnout": 2,
            "q_concentration": 3,
            "q_isolation": 2,
            "q_coping_habits": 0,
            "sleep_hours": 4.5,
            "work_screen_hours": 8.0
        }
    },
    "overwhelmed": {
        "title": "🌧️ Overwhelmed & Isolated",
        "description": "Feelings of hopelessness, low motivation, lack of emotional support, and loss of coping confidence.",
        "data": {
            "q_anhedonia": 3,
            "q_depressed": 3,
            "q_optimism": 0,
            "q_overwhelmed": 4,
            "q_uncontrollable": 4,
            "q_coping_confidence": 1,
            "q_anxious": 3,
            "q_worry": 3,
            "q_somatic": 2,
            "q_sleep_issues": 3,
            "q_fatigue": 3,
            "q_burnout": 3,
            "q_concentration": 3,
            "q_isolation": 4,
            "q_coping_habits": 0,
            "sleep_hours": 5.2,
            "work_screen_hours": 7.0
        }
    }
}

QUESTIONS_METADATA = [
    {
        "id": "q_anhedonia",
        "category": "Mood & Well-being",
        "question": "Little interest or pleasure in doing things you normally enjoy",
        "options": ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Most days", "4 - Nearly every day"]
    },
    {
        "id": "q_depressed",
        "category": "Mood & Well-being",
        "question": "Feeling down, depressed, discouraged, or hopeless",
        "options": ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Most days", "4 - Nearly every day"]
    },
    {
        "id": "q_optimism",
        "category": "Mood & Well-being",
        "question": "Feeling cheerful, optimistic, and looking forward to the future",
        "options": ["0 - Rarely / Never", "1 - Occasionally", "2 - Sometimes", "3 - Frequently", "4 - Consistently"]
    },
    {
        "id": "q_overwhelmed",
        "category": "Stress & Perceived Control",
        "question": "Feeling that difficulties are piling up so high you cannot overcome them",
        "options": ["0 - Never", "1 - Almost Never", "2 - Sometimes", "3 - Fairly Often", "4 - Very Often"]
    },
    {
        "id": "q_uncontrollable",
        "category": "Stress & Perceived Control",
        "question": "Feeling unable to control the important events and stressors in your life",
        "options": ["0 - Never", "1 - Almost Never", "2 - Sometimes", "3 - Fairly Often", "4 - Very Often"]
    },
    {
        "id": "q_coping_confidence",
        "category": "Stress & Perceived Control",
        "question": "Feeling confident in your ability to handle personal problems and challenges",
        "options": ["0 - Not Confident", "1 - Slightly Confident", "2 - Moderately Confident", "3 - Very Confident", "4 - Extremely Confident"]
    },
    {
        "id": "q_anxious",
        "category": "Anxiety & Tension",
        "question": "Feeling nervous, anxious, on edge, or irritable",
        "options": ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Most days", "4 - Nearly every day"]
    },
    {
        "id": "q_worry",
        "category": "Anxiety & Tension",
        "question": "Trouble relaxing or struggling to stop / control repetitive worrying thoughts",
        "options": ["0 - Not at all", "1 - Several days", "2 - More than half the days", "3 - Most days", "4 - Nearly every day"]
    },
    {
        "id": "q_somatic",
        "category": "Anxiety & Tension",
        "question": "Physical stress responses (muscle tension, racing heart, stomach knot, headaches)",
        "options": ["0 - Never", "1 - Rarely", "2 - Sometimes", "3 - Frequently", "4 - Constant / Severe"]
    },
    {
        "id": "q_sleep_issues",
        "category": "Sleep & Recovery",
        "question": "Trouble falling asleep, night awakenings, or waking up completely unrefreshed",
        "options": ["0 - Never", "1 - Rarely", "2 - 1-2 nights/week", "3 - 3-4 nights/week", "4 - Nightly / Chronic"]
    },
    {
        "id": "q_fatigue",
        "category": "Sleep & Recovery",
        "question": "Feeling tired, drained of energy, or experiencing daytime mental exhaustion",
        "options": ["0 - Rarely", "1 - Mild / Occasionally", "2 - Moderate", "3 - Severe / Frequent", "4 - Debilitating"]
    },
    {
        "id": "q_burnout",
        "category": "Work & Cognitive Vitality",
        "question": "Feeling emotionally depleted, burned out, or detached from your work/studies",
        "options": ["0 - Never", "1 - Rarely", "2 - Sometimes", "3 - Often", "4 - Constantly"]
    },
    {
        "id": "q_concentration",
        "category": "Work & Cognitive Vitality",
        "question": "Experiencing brain fog, difficulty concentrating, or trouble making simple decisions",
        "options": ["0 - None", "1 - Slight", "2 - Moderate", "3 - Severe", "4 - Constant"]
    },
    {
        "id": "q_isolation",
        "category": "Social & Coping Resilience",
        "question": "Feeling lonely, misunderstood, or lacking someone you can confide in when stressed",
        "options": ["0 - Well Supported", "1 - Rarely Lonely", "2 - Occasionally Lonely", "3 - Often Isolated", "4 - Severely Isolated"]
    },
    {
        "id": "q_coping_habits",
        "category": "Social & Coping Resilience",
        "question": "Frequency of active stress recovery (mindfulness, walks, hobbies, fitness, venting to peers)",
        "options": ["0 - Never", "1 - Rarely (once a month)", "2 - Sometimes (1-2x/week)", "3 - Regularly (3-4x/week)", "4 - Daily practice"]
    }
]

class MentalHealthPredictor:
    def __init__(self):
        self.model_data = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at {MODEL_PATH}. Generating ML model...")
            try:
                from train_mental_health_model import train_and_export_mental_health_models
                self.model_data = train_and_export_mental_health_models()
                return
            except Exception as e:
                print(f"Notice: Auto-training error ({e}). Using algorithmic psychometric baseline.")
        
        if os.path.exists(MODEL_PATH):
            try:
                self.model_data = joblib.load(MODEL_PATH)
                print(f"Loaded Mental Health & Stress ML model from {MODEL_PATH}")
            except Exception as e:
                print(f"Warning: Failed to load mental health model ({e}), using psychometric fallback.")
                self.model_data = None
        else:
            self.model_data = None

    def _clean_input(self, user_input):
        """Sanitizes and extracts the 17 input features with defensive defaults."""
        def get_int(key, default=1, min_val=0, max_val=4):
            try:
                val = int(user_input.get(key, default))
                return max(min_val, min(max_val, val))
            except (ValueError, TypeError):
                return default

        def get_float(key, default=7.0, min_val=1.0, max_val=24.0):
            try:
                val = float(user_input.get(key, default))
                return max(min_val, min(max_val, val))
            except (ValueError, TypeError):
                return default

        return {
            'q_anhedonia': get_int('q_anhedonia', 1),
            'q_depressed': get_int('q_depressed', 1),
            'q_optimism': get_int('q_optimism', 3),
            'q_overwhelmed': get_int('q_overwhelmed', 1),
            'q_uncontrollable': get_int('q_uncontrollable', 1),
            'q_coping_confidence': get_int('q_coping_confidence', 3),
            'q_anxious': get_int('q_anxious', 1),
            'q_worry': get_int('q_worry', 1),
            'q_somatic': get_int('q_somatic', 1),
            'q_sleep_issues': get_int('q_sleep_issues', 1),
            'q_fatigue': get_int('q_fatigue', 1),
            'q_burnout': get_int('q_burnout', 1),
            'q_concentration': get_int('q_concentration', 1),
            'q_isolation': get_int('q_isolation', 1),
            'q_coping_habits': get_int('q_coping_habits', 2),
            'sleep_hours': get_float('sleep_hours', 7.5, 3.0, 14.0),
            'work_screen_hours': get_float('work_screen_hours', 6.5, 0.5, 18.0)
        }

    def calculate_domain_breakdown(self, data):
        """
        Calculates 6 clinical psychometric domain scores on 0-100 scales:
        1. Emotional Vitality & Mood
        2. Stress Control & Resilience
        3. Anxiety & Nervous Regulation
        4. Sleep & Physical Recovery
        5. Cognitive Vitality & Burnout Resistance
        6. Social Support & Coping Mechanisms
        """
        # 1. Emotional Vitality & Mood (higher is healthier)
        # Positives: optimism (0-4); Negatives: anhedonia (0-4), depressed (0-4)
        mood_raw = data['q_optimism'] + (4 - data['q_anhedonia']) + (4 - data['q_depressed']) # max 12
        mood_score = round((mood_raw / 12.0) * 100, 1)

        # 2. Stress Control & Resilience (higher is better control / lower perceived stress)
        # Positives: coping_confidence; Negatives: overwhelmed, uncontrollable
        stress_ctrl_raw = data['q_coping_confidence'] + (4 - data['q_overwhelmed']) + (4 - data['q_uncontrollable']) # max 12
        stress_control_score = round((stress_ctrl_raw / 12.0) * 100, 1)

        # 3. Anxiety & Nervous Regulation (higher is calmer nervous system)
        anxiety_raw = (4 - data['q_anxious']) + (4 - data['q_worry']) + (4 - data['q_somatic']) # max 12
        anxiety_reg_score = round((anxiety_raw / 12.0) * 100, 1)

        # 4. Sleep & Physical Recovery (higher is more restorative)
        sleep_qual_raw = (4 - data['q_sleep_issues']) + (4 - data['q_fatigue']) # max 8
        sleep_dur = data['sleep_hours']
        sleep_dur_factor = 1.0
        if sleep_dur < 6.0:
            sleep_dur_factor = max(0.4, sleep_dur / 6.0)
        elif sleep_dur > 9.5:
            sleep_dur_factor = max(0.7, 1.0 - (sleep_dur - 9.5) * 0.1)
        sleep_score = round(min(100.0, (sleep_qual_raw / 8.0) * 100 * sleep_dur_factor), 1)

        # 5. Cognitive Vitality & Burnout Resistance (higher is sharper / less burnt out)
        burnout_raw = (4 - data['q_burnout']) + (4 - data['q_concentration']) # max 8
        screen_factor = 1.0
        if data['work_screen_hours'] > 9.0:
            screen_factor = max(0.7, 1.0 - (data['work_screen_hours'] - 9.0) * 0.04)
        burnout_score = round(min(100.0, (burnout_raw / 8.0) * 100 * screen_factor), 1)

        # 6. Social Support & Coping Mechanisms (higher is better buffer)
        coping_raw = (4 - data['q_isolation']) + data['q_coping_habits'] # max 8
        coping_score = round((coping_raw / 8.0) * 100, 1)

        return {
            'emotional_vitality': {
                'name': 'Emotional Vitality & Mood',
                'score': mood_score,
                'status': self._get_domain_status(mood_score),
                'icon': 'fa-solid fa-sun',
                'color': 'amber'
            },
            'stress_resilience': {
                'name': 'Stress Control & Resilience',
                'score': stress_control_score,
                'status': self._get_domain_status(stress_control_score),
                'icon': 'fa-solid fa-shield-heart',
                'color': 'emerald'
            },
            'anxiety_regulation': {
                'name': 'Nervous Regulation & Calm',
                'score': anxiety_reg_score,
                'status': self._get_domain_status(anxiety_reg_score),
                'icon': 'fa-solid fa-wave-square',
                'color': 'teal'
            },
            'sleep_recovery': {
                'name': 'Sleep & Restorative Recovery',
                'score': sleep_score,
                'status': self._get_domain_status(sleep_score),
                'icon': 'fa-solid fa-bed',
                'color': 'indigo'
            },
            'cognitive_vitality': {
                'name': 'Cognitive Vitality & Burnout Resistance',
                'score': burnout_score,
                'status': self._get_domain_status(burnout_score),
                'icon': 'fa-solid fa-brain',
                'color': 'purple'
            },
            'social_coping': {
                'name': 'Social Buffer & Coping Habits',
                'score': coping_score,
                'status': self._get_domain_status(coping_score),
                'icon': 'fa-solid fa-hands-holding-child',
                'color': 'rose'
            }
        }

    def _get_domain_status(self, score):
        if score >= 80:
            return "Robust / Optimal"
        elif score >= 65:
            return "Good / Balanced"
        elif score >= 45:
            return "Moderate / Strained"
        elif score >= 30:
            return "Vulnerable / Needs Attention"
        else:
            return "Severe Depletion"

    def predict(self, user_input):
        """
        Runs ML prediction & psychometric evaluation.
        Returns Mental Health Score (0-100), Stress Score (0-100), domain breakdown,
        risk alerts, psychological strengths, and tailored coping roadmap.
        """
        data = self._clean_input(user_input)
        domains = self.calculate_domain_breakdown(data)

        # ML Model Inference
        feature_cols = [
            'q_anhedonia', 'q_depressed', 'q_optimism', 'q_overwhelmed',
            'q_uncontrollable', 'q_coping_confidence', 'q_anxious', 'q_worry',
            'q_somatic', 'q_sleep_issues', 'q_fatigue', 'q_burnout',
            'q_concentration', 'q_isolation', 'q_coping_habits',
            'sleep_hours', 'work_screen_hours'
        ]
        import pandas as pd
        feature_df = pd.DataFrame([[data[col] for col in feature_cols]], columns=feature_cols)

        if self.model_data and 'mh_model' in self.model_data and 'stress_model' in self.model_data:
            try:
                pred_mh = float(self.model_data['mh_model'].predict(feature_df)[0])
                pred_stress = float(self.model_data['stress_model'].predict(feature_df)[0])
            except Exception as e:
                print(f"ML inference error: {e}, falling back to clinical calculation.")
                pred_mh = self._algorithmic_mh(data, domains)
                pred_stress = self._algorithmic_stress(data, domains)
        else:
            pred_mh = self._algorithmic_mh(data, domains)
            pred_stress = self._algorithmic_stress(data, domains)

        # Enforce bounds
        mental_health_score = round(max(5.0, min(99.0, pred_mh)), 1)
        stress_score = round(max(3.0, min(99.0, pred_stress)), 1)

        # Classifications
        mh_tier = self._classify_mental_health(mental_health_score)
        stress_tier = self._classify_stress(stress_score)

        # Risk Factors & Strengths
        risks = self._extract_risk_factors(data, domains)
        strengths = self._extract_strengths(data, domains)

        # Coping Toolkit & Action Roadmap
        action_plan = self._generate_action_plan(data, stress_score, mental_health_score, risks)

        return {
            'mental_health_score': mental_health_score,
            'stress_score': stress_score,
            'mental_health_tier': mh_tier,
            'stress_tier': stress_tier,
            'domains': domains,
            'radar_labels': [d['name'] for d in domains.values()],
            'radar_values': [d['score'] for d in domains.values()],
            'risk_factors': risks,
            'strengths': strengths,
            'action_plan': action_plan,
            'input_echo': data
        }

    def _algorithmic_mh(self, data, domains):
        # Average domain scores with clinical weighting
        d = domains
        composite = (
            d['emotional_vitality']['score'] * 0.25 +
            d['stress_resilience']['score'] * 0.22 +
            d['anxiety_regulation']['score'] * 0.18 +
            d['sleep_recovery']['score'] * 0.15 +
            d['cognitive_vitality']['score'] * 0.10 +
            d['social_coping']['score'] * 0.10
        )
        return composite

    def _algorithmic_stress(self, data, domains):
        d = domains
        # Invert resilience and calmness domains
        inv_control = 100.0 - d['stress_resilience']['score']
        inv_anxiety = 100.0 - d['anxiety_regulation']['score']
        inv_burnout = 100.0 - d['cognitive_vitality']['score']
        inv_sleep = 100.0 - d['sleep_recovery']['score']
        inv_coping = 100.0 - d['social_coping']['score']

        stress_composite = (
            inv_control * 0.35 +
            inv_anxiety * 0.25 +
            inv_burnout * 0.20 +
            inv_sleep * 0.12 +
            inv_coping * 0.08
        )
        return stress_composite

    def _classify_mental_health(self, score):
        if score >= 85:
            return {
                'title': 'Flourishing & Optimal Well-Being',
                'description': 'High emotional vitality, robust coping buffers, and restorative mental clarity.',
                'badge': 'Optimal',
                'badge_class': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border-emerald-300',
                'color': 'emerald',
                'level': 'optimal'
            }
        elif score >= 70:
            return {
                'title': 'Resilient & Healthy Baseline',
                'description': 'Balanced emotional regulation with solid resilience against everyday stressors.',
                'badge': 'Healthy',
                'badge_class': 'bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300 border-teal-300',
                'color': 'teal',
                'level': 'healthy'
            }
        elif score >= 50:
            return {
                'title': 'Moderate Strain / Mild Vulnerability',
                'description': 'Manageable functioning, but noticeable periods of stress, cognitive fatigue, or low mood.',
                'badge': 'Moderate',
                'badge_class': 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-300',
                'color': 'amber',
                'level': 'moderate'
            }
        elif score >= 35:
            return {
                'title': 'Elevated Distress / Psychological Strain',
                'description': 'Significant emotional exhaustion, anxiety, or depressive symptoms impacting daily energy.',
                'badge': 'Elevated Strain',
                'badge_class': 'bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300 border-orange-300',
                'color': 'orange',
                'level': 'warning'
            }
        else:
            return {
                'title': 'High Distress / Clinical Risk Zone',
                'description': 'Substantial mental overload, pervasive exhaustion, or emotional distress. Proactive care advised.',
                'badge': 'High Risk',
                'badge_class': 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 border-rose-300',
                'color': 'rose',
                'level': 'danger'
            }

    def _classify_stress(self, score):
        if score <= 25:
            return {
                'title': 'Low / Serene Stress Level',
                'description': 'Calm physiological state with healthy autonomic regulation and low sympathetic arousal.',
                'badge': 'Serene / Low',
                'badge_class': 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300 border-emerald-300',
                'color': 'emerald',
                'level': 'optimal'
            }
        elif score <= 50:
            return {
                'title': 'Moderate / Manageable Stress',
                'description': 'Normal everyday situational demands. Recoverable with ordinary sleep and downtime.',
                'badge': 'Manageable',
                'badge_class': 'bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-teal-300 border-teal-300',
                'color': 'teal',
                'level': 'healthy'
            }
        elif score <= 75:
            return {
                'title': 'Elevated / High Chronic Stress',
                'description': 'Persistent nervous activation, mental overload, or emotional tension. Active decompression needed.',
                'badge': 'High Stress',
                'badge_class': 'bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300 border-amber-300',
                'color': 'amber',
                'level': 'warning'
            }
        else:
            return {
                'title': 'Severe / Critical Acute Overwhelm',
                'description': 'Acute autonomic exhaustion or burnout threshold. High risk of cognitive depletion and physical toll.',
                'badge': 'Critical / Severe',
                'badge_class': 'bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300 border-rose-300',
                'color': 'rose',
                'level': 'danger'
            }

    def _extract_risk_factors(self, data, domains):
        risks = []
        if data['q_overwhelmed'] >= 3 or data['q_uncontrollable'] >= 3:
            risks.append({
                'title': 'High Perceived Helplessness / Loss of Control',
                'detail': 'Feelings that difficulties are piling up beyond coping capacity. Common driver of chronic anxiety.',
                'severity': 'high',
                'icon': 'fa-solid fa-tornado'
            })

        if data['q_somatic'] >= 3:
            risks.append({
                'title': 'Severe Somatic Stress Manifestation',
                'detail': 'Physical symptoms like muscle clenching, palpitations, and gastrointestinal tension indicating sympathetic overload.',
                'severity': 'high',
                'icon': 'fa-solid fa-heart-crack'
            })

        if data['q_sleep_issues'] >= 3 or data['sleep_hours'] < 6.0:
            risks.append({
                'title': 'Circadian & Sleep Disruption',
                'detail': f"Getting {data['sleep_hours']}h of sleep with frequent night disruptions prevents restorative neurochemical reset.",
                'severity': 'high',
                'icon': 'fa-solid fa-moon'
            })

        if data['q_burnout'] >= 3 or (data['work_screen_hours'] >= 10.0 and data['q_concentration'] >= 2):
            risks.append({
                'title': 'Occupational / Cognitive Burnout',
                'detail': f"Extended screen/work time ({data['work_screen_hours']}h/day) coupled with brain fog and emotional detachment.",
                'severity': 'medium',
                'icon': 'fa-solid fa-fire'
            })

        if data['q_anhedonia'] >= 3 or data['q_depressed'] >= 3:
            risks.append({
                'title': 'Persistent Low Mood & Anhedonia',
                'detail': 'Difficulty experiencing pleasure in usual interests alongside prolonged low spirits.',
                'severity': 'high',
                'icon': 'fa-solid fa-cloud-rain'
            })

        if data['q_isolation'] >= 3:
            risks.append({
                'title': 'Social Isolation & Limited Support Anchor',
                'detail': 'Lack of emotionally safe outlets to vocalize challenges significantly multiplies perceived stress.',
                'severity': 'medium',
                'icon': 'fa-solid fa-user-xmark'
            })

        if data['q_coping_habits'] <= 1:
            risks.append({
                'title': 'Deficit in Active Decompression Habits',
                'detail': 'Minimal time dedicated to somatic reset, cardiovascular exercise, or intentional restorative hobbies.',
                'severity': 'medium',
                'icon': 'fa-solid fa-battery-empty'
            })

        return risks

    def _extract_strengths(self, data, domains):
        strengths = []
        if data['q_coping_confidence'] >= 3:
            strengths.append({
                'title': 'Strong Problem-Solving Self-Efficacy',
                'detail': 'Confidence in your personal capability to overcome life challenges is a powerful psychological resilience anchor.'
            })

        if data['q_optimism'] >= 3:
            strengths.append({
                'title': 'Positive Emotional Outlook',
                'detail': 'Consistent optimism buffers against chronic depressive patterns and supports adaptive coping.'
            })

        if data['q_coping_habits'] >= 3:
            strengths.append({
                'title': 'Proactive Restorative Habits',
                'detail': 'Regular engagement in exercise, mindfulness, or hobbies provides regular parasympathetic activation.'
            })

        if data['q_isolation'] <= 1:
            strengths.append({
                'title': 'Healthy Social Support Network',
                'detail': 'Having trusted peers or family to share burdens reduces neurological cortisol spikes during crises.'
            })

        if data['sleep_hours'] >= 7.0 and data['q_sleep_issues'] <= 1:
            strengths.append({
                'title': 'Restorative Sleep Foundation',
                'detail': 'Consistently getting 7+ hours of quality sleep protects prefrontal cortex emotional regulation.'
            })

        if not strengths:
            strengths.append({
                'title': 'Awareness & Proactive Engagement',
                'detail': 'Taking this assessment demonstrates self-awareness and an intention to prioritize your mental wellbeing.'
            })

        return strengths

    def _generate_action_plan(self, data, stress_score, mh_score, risks):
        """Builds a 3-phase evidence-based personalized mental health & stress roadmap."""
        # Immediate Relief Toolkit
        immediate_toolkit = [
            {
                'title': '4-7-8 Parasympathetic Vagus Nerve Reset',
                'desc': 'Inhale quietly through nose for 4s, hold breath for 7s, exhale completely through mouth with a whoosh for 8s. Repeat 4 cycles to immediately trigger parasympathetic braking.',
                'tag': 'Immediate Calming'
            },
            {
                'title': '5-4-3-2-1 Sensory Grounding Technique',
                'desc': 'Acknowledge 5 things you can see, 4 you can feel, 3 you can hear, 2 you can smell, and 1 you can taste to pull your nervous system out of an anxious spiral.',
                'tag': 'Acute Anxiety Break'
            },
            {
                'title': 'Somatic Sigh & Progressive Jaw/Shoulder Drop',
                'desc': 'Take two quick sniffs in through the nose, followed by one long unforced sigh out. Noticeably drop your shoulders away from your ears and unclamp your jaw.',
                'tag': 'Physical Tension Release'
            }
        ]

        # Phase 1: Days 1-7 (Nervous System Stabilization)
        phase1_items = [
            'Establish an unbroken 60-minute digital curfew before sleep to restore melatonin production.',
            'Institute micro-decompression intervals: step away from screens for 5 minutes every 50 minutes of focused work.',
            'Hydrate immediately upon waking with 500ml water and get 10 minutes of natural outdoor light into your eyes.'
        ]
        if data['q_sleep_issues'] >= 2:
            phase1_items.append('Keep your sleep and wake times strictly consistent (+/- 30 mins), even on weekends.')
        if data['q_somatic'] >= 2:
            phase1_items.append('Perform a 10-minute Progressive Muscle Relaxation (PMR) session prior to entering bed.')

        # Phase 2: Weeks 2-4 (Cognitive Reframing & Boundaries)
        phase2_items = [
            'Cognitive Brain Dump: spend 5 minutes every evening writing tomorrow’s top 3 tasks on paper to empty working memory.',
            'Practice the "Circle of Control" audit: identify which current stressors are actionable vs which require emotional acceptance.',
            'Implement firm work boundary hours: disable non-critical work notifications past 7:00 PM.'
        ]
        if data['q_isolation'] >= 2:
            phase2_items.append('Schedule at least one intentional weekly connection (coffee, phone call, shared walk) with an uplifting friend or family member.')
        if data['q_coping_habits'] <= 1:
            phase2_items.append('Commit to 20 minutes of brisk outdoor walking or moderate cardio 4 times per week to boost brain-derived neurotrophic factor (BDNF).')

        # Phase 3: Month 2+ (Long-Term Psychological Resilience & Sustainability)
        phase3_items = [
            'Incorporate daily mindfulness, journal prompts, or gratitude reflection to strengthen emotional cognitive flexibility.',
            'Conduct a monthly burnout audit to identify energy vampires and protect your calendar bandwidth.',
            'Engage in a creative or playful hobby purely for intrinsic enjoyment rather than productivity.'
        ]

        # Crisis Resources / Disclaimer
        resources = [
            {'name': 'National Suicide Prevention & Crisis Lifeline (US/Canada)', 'contact': 'Dial 988 (Available 24/7, Free & Confidential)'},
            {'name': 'Crisis Text Line', 'contact': 'Text HOME to 741741 to connect with a Crisis Counselor'},
            {'name': 'India National Mental Health Helpline (KIRAN)', 'contact': '1800-599-0019 (24/7 Toll-Free)'},
            {'name': 'UK NHS Mental Health Services', 'contact': 'Dial 111 (Mental Health Option)'},
            {'name': 'International Crisis Resources (Befrienders)', 'contact': 'Visit https://www.befrienders.org for global local hotlines'}
        ]

        return {
            'immediate_toolkit': immediate_toolkit,
            'phase1': {
                'title': 'Phase 1: Days 1-7 — Autonomic Decompression & Sleep Stabilization',
                'actions': phase1_items
            },
            'phase2': {
                'title': 'Phase 2: Weeks 2-4 — Cognitive Boundaries & Cortisol Regulation',
                'actions': phase2_items
            },
            'phase3': {
                'title': 'Phase 3: Month 2+ — Sustainable Psychological Fitness & Flourishing',
                'actions': phase3_items
            },
            'resources': resources,
            'disclaimer': 'Vitalis Mental Health Predictor is an educational self-reflection screening and machine learning tool, not a diagnostic clinical evaluation. If you are experiencing persistent distress, panic, or clinical symptoms, please consult a licensed psychologist or healthcare provider.'
        }
