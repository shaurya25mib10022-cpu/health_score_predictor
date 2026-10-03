"""
mental_health_cli.py
Interactive Command-Line Interface for the Mental Health & Stress Score Predictor.
Asks the user evidence-based screening questions and calculates:
1. Mental Health Score (0 - 100)
2. Stress Score (0 - 100)
3. Domain Breakdown (Mood, Stress Resilience, Anxiety, Sleep, Burnout, Coping)
4. Key Red Flags, Protective Strengths, and Personalized Coping Steps
"""

import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from mental_health_engine import MentalHealthPredictor, QUESTIONS_METADATA

def print_banner():
    banner = r"""
==========================================================================
    🧠  AI MENTAL HEALTH & STRESS SCORE PREDICTOR  🌿
    Evidence-based Psychometric & Machine Learning Assessment
==========================================================================
This assessment evaluates your psychological well-being, perceived stress,
anxiety levels, sleep impact, and burnout indicators.
For each question, select the rating from 0 to 4 that best matches you
over the past 2 weeks.
--------------------------------------------------------------------------
"""
    print(banner)

def ask_question(q_meta, idx, total):
    print(f"\n[{idx}/{total}] Category: {q_meta['category']}")
    print(f"👉 {q_meta['question']}")
    for opt in q_meta['options']:
        print(f"   [{opt[0]}] {opt[3:]}")
    
    while True:
        try:
            choice = input("Enter choice (0-4): ").strip()
            if choice in ['0', '1', '2', '3', '4']:
                return int(choice)
            print("⚠️ Please enter a valid number between 0 and 4.")
        except (KeyboardInterrupt, EOFError):
            print("\nAssessment cancelled.")
            sys.exit(0)

def ask_float(prompt, default, min_val, max_val):
    while True:
        try:
            raw = input(f"{prompt} [Default: {default}]: ").strip()
            if not raw:
                return default
            val = float(raw)
            if min_val <= val <= max_val:
                return val
            print(f"⚠️ Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("⚠️ Please enter a valid decimal number.")
        except (KeyboardInterrupt, EOFError):
            print("\nAssessment cancelled.")
            sys.exit(0)

def main():
    print_banner()
    predictor = MentalHealthPredictor()
    
    user_data = {}
    total_q = len(QUESTIONS_METADATA) + 2
    
    for i, q in enumerate(QUESTIONS_METADATA, 1):
        user_data[q['id']] = ask_question(q, i, total_q)
        
    print(f"\n[{len(QUESTIONS_METADATA)+1}/{total_q}] Daily Sleep Context:")
    user_data['sleep_hours'] = ask_float("Average hours of sleep per night (3.0 - 14.0)", 7.0, 3.0, 14.0)
    
    print(f"\n[{total_q}/{total_q}] Daily Screen / Intense Focus Context:")
    user_data['work_screen_hours'] = ask_float("Average daily screen or work hours (1.0 - 18.0)", 7.5, 1.0, 18.0)
    
    print("\n⏳ Processing responses through Psychometric & ML Engine...")
    result = predictor.predict(user_data)
    
    mh_score = result['mental_health_score']
    st_score = result['stress_score']
    mh_tier = result['mental_health_tier']
    st_tier = result['stress_tier']
    
    print("\n" + "=" * 74)
    print("                 📊 YOUR ASSESSMENT RESULTS")
    print("=" * 74)
    print(f"🧠 MENTAL HEALTH SCORE :  {mh_score:.1f} / 100  [{mh_tier['title'].upper()}]")
    print(f"   → Status: {mh_tier['description']}")
    print("-" * 74)
    print(f"⚡ STRESS SCORE        :  {st_score:.1f} / 100  [{st_tier['title'].upper()}]")
    print(f"   → Status: {st_tier['description']}")
    print("=" * 74)
    
    print("\n📈 Granular Domain Breakdown (0 - 100):")
    for key, dom in result['domains'].items():
        bar_len = int(dom['score'] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  • {dom['name']:<40} [{bar}] {dom['score']:>5.1f}% ({dom['status']})")
        
    if result['risk_factors']:
        print("\n⚠️ Identified Risk Red Flags:")
        for r in result['risk_factors']:
            print(f"  [!] {r['title']}: {r['detail']}")
            
    if result['strengths']:
        print("\n🌟 Key Psychological Strengths:")
        for s in result['strengths']:
            print(f"  [✓] {s['title']}: {s['detail']}")
            
    print("\n🧘 Immediate Coping Toolkit:")
    for tool in result['action_plan']['immediate_toolkit']:
        print(f"  • [{tool['tag']}] {tool['title']}:")
        print(f"    {tool['desc']}")
        
    print("\n📅 Recommended 3-Phase Action Roadmap:")
    p1 = result['action_plan']['phase1']
    print(f"\n  [ {p1['title']} ]")
    for item in p1['actions']:
        print(f"    - {item}")
        
    p2 = result['action_plan']['phase2']
    print(f"\n  [ {p2['title']} ]")
    for item in p2['actions']:
        print(f"    - {item}")
        
    p3 = result['action_plan']['phase3']
    print(f"\n  [ {p3['title']} ]")
    for item in p3['actions']:
        print(f"    - {item}")
        
    print("\n" + "=" * 74)
    print("📞 Support Helpline Resources:")
    for res in result['action_plan']['resources']:
        print(f"  • {res['name']}: {res['contact']}")
    print("-" * 74)
    print(f"ℹ️ Disclaimer: {result['action_plan']['disclaimer']}")
    print("=" * 74 + "\n")

if __name__ == '__main__':
    main()
