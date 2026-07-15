# report_generator.py
# GenAI Report Generator — Google Gemini 1.5 Flash
# Developer: Hamna Munir | Supervisor: Sir Nadeem

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-latest"]

def _get_key():
    try:
        k = st.secrets.get("GEMINI_API_KEY")
        if k: return k
    except: pass
    return os.getenv("GEMINI_API_KEY")

def _call_gemini(prompt):
    key = _get_key()
    if not key:
        return None, "No API key found"
    for model in MODELS:
        try:
            from google import genai
            client   = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model, contents=prompt)
            return response.text.strip(), None
        except Exception as e:
            err = str(e)
            if "404" in err or "not found" in err.lower(): continue
            elif "429" in err or "QUOTA" in err.upper():
                return None, "quota_exceeded"
            else: continue
    return None, "All models failed"

# ══════════════════════════════════════════════════════
# REPORT 1 — Today's Class Summary
# ══════════════════════════════════════════════════════
def generate_class_summary(summary, class_data=None):
    students_info = ""
    if class_data:
        focused    = [r for r in class_data if r.get("score",0) >= 60]
        distracted = [r for r in class_data if r.get("score",0) < 35]
        students_info = f"""
Student breakdown:
- Focused students: {[r.get('name','?') for r in focused]}
- Distracted students: {[r.get('name','?') for r in distracted]}
- Total analyzed: {len(class_data)}"""

    prompt = f"""You are an expert AI education analyst. 
Analyze this classroom session data and write a professional summary report.

Session Data:
- Total readings: {summary.get('total_readings', 0)}
- Average engagement: {summary.get('avg_engagement', 0)}/100
- Dominant state: {summary.get('dominant_state', 'N/A')}
- Dominant emotion: {summary.get('dominant_emotion', 'N/A')}
- Session duration: {summary.get('duration_min', 0)} minutes
- Total alerts fired: {summary.get('alert_count', 0)}
{students_info}

Write a professional 4-sentence class summary:
1. Overall class performance assessment
2. Engagement pattern observation  
3. Emotional climate of the class
4. Key recommendation for the teacher

Be specific, warm, and actionable. No bullet points."""

    result, error = _call_gemini(prompt)
    if result: return result
    return _fallback_summary(summary)

# ══════════════════════════════════════════════════════
# REPORT 2 — Students Who Need Improvement
# ══════════════════════════════════════════════════════
def generate_improvement_report(summary, class_data=None, agent_summary=None):
    at_risk = []
    if class_data:
        at_risk = [r for r in class_data if r.get("score",0) < 40]

    at_risk_info = "\n".join([
        f"- {r.get('name','Student')}: {r.get('score',0)}% ({r.get('state','?')}), emotion: {r.get('emotion','?')}"
        for r in at_risk
    ]) if at_risk else "No severely distracted students detected"

    agent_info = ""
    if agent_summary:
        agent_info = f"""
Agent detected:
- Total alerts: {agent_summary.get('total_alerts', 0)}
- Flagged students: {agent_summary.get('flagged_count', 0)}
- Chronic cases: {agent_summary.get('chronic_count', 0)}"""

    prompt = f"""You are a supportive AI education coach helping teachers identify students who need extra support.

Students needing attention:
{at_risk_info}
{agent_info}

Class average: {summary.get('avg_engagement', 0)}/100
Session duration: {summary.get('duration_min', 0)} minutes

Write a compassionate, professional 4-sentence report:
1. Identify which students need immediate attention
2. Describe the pattern of disengagement observed
3. Suggest a specific intervention strategy for the teacher
4. Give one positive encouragement for both teacher and students

Be kind, specific, and solution-focused. No bullet points."""

    result, error = _call_gemini(prompt)
    if result: return result
    return _fallback_improvement(summary, at_risk)

# ══════════════════════════════════════════════════════
# REPORT 3 — Teaching Effectiveness Report
# ══════════════════════════════════════════════════════
def generate_effectiveness_report(summary, agent_summary=None):
    agent_info = ""
    if agent_summary:
        agent_info = f"""
Behavioral patterns detected:
- Alerts fired: {agent_summary.get('total_alerts', 0)}
- Class average score: {agent_summary.get('class_avg', 0)}/100
- Session duration: {agent_summary.get('session_min', 0)} min"""

    prompt = f"""You are an expert teaching effectiveness analyst using AI classroom data.

Session metrics:
- Average engagement: {summary.get('avg_engagement', 0)}/100
- Dominant student state: {summary.get('dominant_state', 'N/A')}
- Dominant emotion: {summary.get('dominant_emotion', 'N/A')}
- Session duration: {summary.get('duration_min', 0)} minutes
- Total readings analyzed: {summary.get('total_readings', 0)}
{agent_info}

Write a professional 4-sentence teaching effectiveness report:
1. Rate the overall teaching session effectiveness (score/10)
2. Describe what was working well based on engagement data
3. Identify the biggest area for improvement
4. Give one specific, actionable strategy for the next class

Be professional, data-driven, and constructive. No bullet points."""

    result, error = _call_gemini(prompt)
    if result: return result
    return _fallback_effectiveness(summary)

# ══════════════════════════════════════════════════════
# FALLBACKS — when Gemini unavailable
# ══════════════════════════════════════════════════════
def _fallback_summary(summary):
    avg   = summary.get('avg_engagement', 0)
    state = summary.get('dominant_state', 'Moderate Focus')
    dur   = summary.get('duration_min', 0)
    if avg >= 70:
        perf = f"Excellent session with {avg}/100 average engagement — students were highly focused throughout."
    elif avg >= 45:
        perf = f"Moderate session with {avg}/100 engagement — most students maintained reasonable attention."
    else:
        perf = f"Challenging session with {avg}/100 engagement — significant attention issues were detected."
    return (f"{perf} The dominant student state was {state}, "
            f"observed over {dur} minutes of classroom monitoring. "
            f"Emotional climate appeared stable with occasional fluctuations in attention levels. "
            f"Consider incorporating more interactive activities to boost engagement in the next session. "
            f"\n\n_(AI quota exceeded — smart fallback report)_")

def _fallback_improvement(summary, at_risk):
    count = len(at_risk)
    if count == 0:
        return ("All students showed acceptable engagement levels — no immediate intervention needed. "
                "Continue with the current teaching approach as it appears to be working well. "
                "Monitor students showing moderate scores for any declining trends. "
                "A brief check-in at the start of next class can help maintain this positive momentum.")
    names = ", ".join([r.get('name','Student') for r in at_risk[:3]])
    return (f"{count} student(s) ({names}) showed low engagement during this session. "
            f"These students averaged below 35% focus score, indicating significant distraction. "
            f"Consider a brief one-on-one check-in and seat repositioning closer to the board. "
            f"Breaking the next session into shorter 15-minute focused blocks may help these students re-engage."
            f"\n\n_(AI quota exceeded — smart fallback report)_")

def _fallback_effectiveness(summary):
    avg = summary.get('avg_engagement', 0)
    score = round(avg / 10, 1)
    return (f"Teaching effectiveness rated at {score}/10 based on student engagement data. "
            f"Average focus of {avg}/100 suggests {'strong' if avg>=60 else 'moderate' if avg>=40 else 'low'} "
            f"student attention during instruction. "
            f"{'The pacing and content delivery appear well-matched to student capacity.' if avg>=60 else 'Consider varying the instructional pace and adding visual aids to boost retention.'} "
            f"For the next session, implement the 10-2 method: 10 minutes of instruction followed by 2 minutes of student reflection."
            f"\n\n_(AI quota exceeded — smart fallback report)_")

# ── Keep backward compatibility
def generate_class_report(summary):
    return generate_class_summary(summary)