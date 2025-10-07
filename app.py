# app.py
import os
import re
import json
import regex as re
import streamlit as st
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# --- Gemini client (Google GenAI SDK) ---
try:
    from google import genai
    from google.genai import types
except Exception as e:
    st.error("Missing google-genai package. Install with: pip install google-genai")
    raise

# Read API key from environment (recommended)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # If not set, we still attempt to initialize client without explicit api_key
    # (the SDK can pick it up from env). We'll still warn user.
    st.warning("GEMINI_API_KEY is not set as an environment variable. Set it for best results.")

# Initialize client (passing api_key is optional if env var is set)
try:
    client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else genai.Client()
except Exception as e:
    st.error("Failed to create Gemini client. Check GEMINI_API_KEY and internet connection.")
    raise

# --- Helpers ---
def extract_json_embedded(text: str) -> Optional[dict]:
    """
    Try to extract the first JSON object found in text robustly.
    """
    # Find curly-brace JSON blocks
    matches = re.findall(r"(\{(?:[^{}]|(?R))*\})", text, flags=re.DOTALL)
    if not matches:
        # fallback: look for triple-backtick JSON
        code_matches = re.findall(r"```(?:json)?\s*(\{(?:.|\n)*?\})\s*```", text, flags=re.DOTALL)
        matches = code_matches
    if not matches:
        return None
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            continue
    return None

def call_gemini_for_json(prompt: str, model: str = "gemini-2.5-flash", thinking_budget: int = 0) -> dict:
    """
    Sends a prompt asking the LLM to output JSON only, and returns the parsed dict.
    """
    # Ask Gemini to produce JSON only
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget)
            ),
        )
    except Exception as e:
        st.error(f"Gemini API request failed: {e}")
        return {}

    raw = response.text if hasattr(response, "text") else str(response)
    parsed = extract_json_embedded(raw)
    if parsed is None:
        # Try a fallback: return raw text under 'raw' key
        return {"raw": raw}
    return parsed

# --- Sentiment / mood classification prompt ---
MOOD_CLASSIFICATION_PROMPT_TEMPLATE = """
You are an assistant that MUST return JSON only (no extra commentary).
Analyze the user's message and return a JSON object with these fields:
- mood: one of ["very_negative","negative","neutral","positive","very_positive"]
- sentiment_score: float between -1.0 (very negative) and 1.0 (very positive)
- tags: array of 0..6 short keyword strings describing content (e.g. "lonely", "exam stress", "sleep", "procrastination")
- risk: one of ["none","low","moderate","high"] where "high" signals self-harm or imminent danger
- short_summary: one short (<=20 words) sentence summarizing the user's emotional state

User message:
\"\"\"{message}\"\"\"
Return JSON only.
"""

# --- Response generation prompt template ---
REPLY_GENERATION_PROMPT_TEMPLATE = """
You are a compassionate mental health chatbot for students. Use the given mood analysis to craft a supportive response.
Return JSON only with keys:
- reply: a short empathetic response (2-5 sentences). Validate the user's feelings; avoid giving clinical diagnosis.
- suggestions: an array of up to 4 practical coping tips (each 1 sentence).
- breathing_exercise: a short 3-step breathing or grounding exercise the user can do in 1-3 minutes.
- tone: single-word tone descriptor you used (e.g., "gentle", "encouraging").

Context:
User message: \"\"\"{message}\"\"\"
Mood analysis (JSON): {mood_json}

Behavior rules:
- Be empathetic, non-judgmental, and concise.
- If mood analysis 'risk' == 'high', do NOT give instructions for self-harm; instead, keep message supportive AND explicitly advise immediate contact of local emergency services or hotlines.
- Do NOT attempt clinical diagnosis or provide medical instructions.
Return JSON only.
"""

# --- Static relaxation tips (fallback if needed) ---
FALLBACK_RELAXATION = {
    "3_breaths": "Sit comfortably. Inhale for 4s, hold 2s, exhale 6s. Repeat 3 times.",
    "grounding_5-4-3-2-1": "Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste or notice.",
    "progressive_muscle": "Tense and relax muscle groups from feet up for 1-2 minutes."
}

# --- Streamlit UI setup ---
st.set_page_config(page_title="Student Mental Health Companion", layout="wide")
st.title("🧡 Mental Health Companion (Demo)")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"sender": "user"/"bot", "text": "..."}
if "last_mood" not in st.session_state:
    st.session_state.last_mood = {}

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Chat")
    user_input = st.text_input("Say something (private). Example: 'I'm so stressed about exams and can't sleep.'", key="input")
    if st.button("Send") and user_input.strip():
        # Record user message
        st.session_state.history.append({"sender": "user", "text": user_input})
        # 1) Classify mood
        classification_prompt = MOOD_CLASSIFICATION_PROMPT_TEMPLATE.format(message=user_input)
        mood_result = call_gemini_for_json(classification_prompt)
        # If Gemini returned raw text only or parsing failed, set a fallback classification
        if not mood_result or "raw" in mood_result:
            # fallback naive heuristic: check for keywords
            txt = user_input.lower()
            if any(k in txt for k in ["suicid", "kill myself", "end my life", "die by"]):
                mood_result = {"mood": "very_negative", "sentiment_score": -0.95, "tags": ["self-harm"], "risk": "high", "short_summary": "Possible self-harm language detected"}
            elif any(k in txt for k in ["anx", "anxi", "panic", "stressed"]):
                mood_result = {"mood": "negative", "sentiment_score": -0.5, "tags": ["anxiety", "stress"], "risk": "low", "short_summary": "Displays anxiety/stress"}
            else:
                mood_result = {"mood": "neutral", "sentiment_score": 0.0, "tags": [], "risk": "none", "short_summary": "Could not confidently parse mood"}
        st.session_state.last_mood = mood_result

        # 2) Generate empathetic reply
        reply_prompt = REPLY_GENERATION_PROMPT_TEMPLATE.format(
            message=user_input, mood_json=json.dumps(mood_result)
        )
        reply_result = call_gemini_for_json(reply_prompt)
        # fallback if parsing fails
        if not reply_result or "raw" in reply_result:
            reply_text = (
                "Thanks for sharing — that sounds really tough. "
                "It might help to try a short grounding exercise now. "
                "If you're in crisis or thinking about harming yourself, please contact local emergency services."
            )
            suggestions = [
                "Try 3 deep breaths (inhale 4s, hold 2s, exhale 6s).",
                "Write one small next step and do that first.",
                "Text or call a friend or counselor and say you need support."
            ]
            breathing = FALLBACK_RELAXATION["3_breaths"]
        else:
            reply_text = reply_result.get("reply", "")
            suggestions = reply_result.get("suggestions", [])
            breathing = reply_result.get("breathing_exercise", "")

        # Append bot reply
        st.session_state.history.append({"sender": "bot", "text": reply_text, "suggestions": suggestions, "breathing": breathing})

    # Show chat history
    for entry in st.session_state.history:
        if entry["sender"] == "user":
            st.markdown(f"**You:** {entry['text']}")
        else:
            st.markdown(f"**Companion:** {entry['text']}")
            if entry.get("suggestions"):
                st.markdown("**Suggestions:**")
                for s in entry["suggestions"]:
                    st.markdown(f"- {s}")
            if entry.get("breathing"):
                st.markdown(f"**Quick exercise:** {entry['breathing']}")

    # Download conversation
    if st.button("Download transcript"):
        lines = []
        for e in st.session_state.history:
            who = "You" if e["sender"] == "user" else "Companion"
            lines.append(f"{who}: {e['text']}")
        txt = "\n".join(lines)
        st.download_button("Download .txt", txt, file_name="chat_transcript.txt", mime="text/plain")

with col2:
    st.header("Mood & Quick Tools")
    mood = st.session_state.get("last_mood", {})
    if mood:
        st.subheader("Detected mood")
        st.write(f"**Mood label:** {mood.get('mood')}")
        st.write(f"**Sentiment score:** {mood.get('sentiment_score')}")
        st.write(f"**Tags:** {', '.join(mood.get('tags',[]))}")
        st.write(f"**Summary:** {mood.get('short_summary')}")
        if mood.get("risk") == "high":
            st.error("⚠️ High risk detected. If you are in immediate danger or experiencing self-harm thoughts, contact local emergency services now.")
            st.markdown(
                "> This app is **not** a substitute for professional care. "
                "Add your local crisis hotlines to the `EMERGENCY_CONTACTS` config in the code."
            )
    else:
        st.write("No mood detected yet. Send a message to begin.")

    st.markdown("---")
    st.subheader("Quick relaxation")
    st.write("- 3 deep breaths: inhale 4s — hold 2s — exhale 6s (repeat 3x)")
    st.write("- Grounding: 5-4-3-2-1 (see UI)")
    st.write("- Progressive muscle relaxation: tension/release from feet to head")

    st.markdown("---")
    st.subheader("Customize")
    model_choice = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)
    st.caption("Using Gemini models for both classification and generation. Update model as needed.")

st.markdown("---")
st.caption("Disclaimer: This is a demo tool. It is not a replacement for professional mental health services. Do not rely on it for emergencies.")
