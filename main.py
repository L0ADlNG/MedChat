import streamlit as st
import google.generativeai as genai
from typing import Generator
from google.generativeai.types import HarmCategory, HarmBlockThreshold

st.set_page_config(page_icon="🩺", layout="wide", page_title="MedChat")

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Medical Safety Configuration
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

SYSTEM_PROMPT = """You are MedChat, a clinical AI assistant. Rules:
1. Provide evidence-based medical information
2. Cite public sources using full institutional names
3. Never mention personal health information
4. Follow this citation format: (Source: [Organization Full Name])
5. Give causes, symptoms, effects, medications or any other information when possible
6. Whenever answer is better described in a tabular form, make a simple table"""

DISCLAIMER = """**Clinical Safety Protections**
- Source citations preserved
- No personal health data collection
- Always verify with healthcare providers"""

def icon(emoji: str):
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🩺")
st.subheader("MedChat", divider="blue", anchor=False)
st.sidebar.title("🩺MedChat")
st.sidebar.markdown(DISCLAIMER)

@st.cache_resource
def load_model():
    try:
        return genai.GenerativeModel(
            model_name='gemini-1.5-pro-latest',
            safety_settings=SAFETY_SETTINGS,
            system_instruction=SYSTEM_PROMPT
        )
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        st.stop()

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "I'm MedChat, a clinical AI assistant. How can I help you today?"
    }]

for message in st.session_state.messages:
    avatar = '🩺' if message["role"] == "assistant" else '👤'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def format_history():
    """Converts session messages to Gemini-compatible format"""
    history = []
    for msg in st.session_state.messages:
        role = 'user' if msg["role"] == "user" else 'model'
        history.append({
            'role': role,
            'parts': [{'text': msg["content"]}]
        })
    return history

def generate_gemini_response(prompt: str) -> Generator[str, None, None]:
    try:
        # Start chat with formatted history
        chat = model.start_chat(history=format_history())
        response = chat.send_message(
            prompt,
            stream=True,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7,
                top_p=0.95
            )
        )
        
        buffer = ""
        for chunk in response:
            text = chunk.text
            if text and text[-1] in {'.', '!', '?', ':', '\n'}:
                yield buffer + text
                buffer = ""
            else:
                buffer += text
                
        if buffer:
            yield buffer
            
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar='👤'):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant", avatar="🩺"):
            warning = st.warning("Analyzing medical query...")
            
            full_response = ""
            response_generator = generate_gemini_response(prompt)
            
            response_placeholder = st.empty()
            for chunk in response_generator:
                full_response += chunk
                response_placeholder.markdown(full_response)
            
            response_placeholder.markdown(full_response)
            
            if "consult" not in full_response.lower():
                st.toast("Remember: Always verify with a healthcare provider", icon="⚠️")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
            warning.empty()
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "System error - please try again"
        })

st.info("ℹ️ Consult a real doctor for medical advice. Use AI only for reference.")
