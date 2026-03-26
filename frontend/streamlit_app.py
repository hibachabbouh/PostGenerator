import os
import requests
import streamlit as st
from typing import Dict, Any
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
STYLES = ["✨ Motivational", "😂 Funny", "🎨 Aesthetic", "📝 General", "🔥 Trendy"]

def _render_header():
    st.set_page_config(page_title="InstaGen AI", page_icon="📸", layout="wide")
    st.markdown("""
        <style>
        .main { background-color: #fafafa; }
        .stButton>button {
            border-radius: 20px;
            background: linear-gradient(45deg, #f09433 0%,#e6683c 25%,#dc2743 50%,#cc2366 75%,#bc1888 100%);
            color: white;
            border: none;
        }
        .caption-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #dc2743;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📸 InstaGen AI")
    st.caption("Craft the perfect caption using RAG-powered intelligence.")

def main():
    _render_header()
    if "history" not in st.session_state:
        st.session_state.history = []
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_url = st.text_input("Backend URL", value=DEFAULT_API_BASE_URL)
        timeout = st.slider("Timeout (s)", 5, 60, 30)
        st.divider()
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Post Details")
        with st.container(border=True):
            topic = st.text_area("What is your post about?", placeholder="e.g., A sunset at the beach in Tunisia", height=150)
            style_choice = st.selectbox("Choose a Vibe", STYLES)
            include_hashtags = st.toggle("Include Hashtags", value=True)
            
            if st.button("Generate Magic ✨", use_container_width=True):
                if not topic:
                    st.error("Please describe your post!")
                else:
                    with st.spinner("Writing your caption..."):
                        try:
                            payload = {"topic": topic, "style": style_choice, "hashtags": include_hashtags}
                            response = requests.post(f"{api_url}/api/generate", json=payload, timeout=timeout)
                            response.raise_for_status()
                            caption = response.json().get("caption", "")
                            
                            st.session_state.current_caption = caption
                            st.session_state.history.append({"topic": topic, "caption": caption})
                        except Exception as e:
                            st.error(f"Connection failed: {e}")

    with col2:
        st.subheader("Result")
        if "current_caption" in st.session_state:
            st.markdown(f"""
                <div class="caption-card">
                    {st.session_state.current_caption}
                </div>
            """, unsafe_allow_html=True)
            st.button("Copy to Clipboard (Simulated)")
        else:
            st.info("Your generated caption will appear here.")

    if st.session_state.history:
        st.divider()
        st.subheader("Recent Generations")
        for item in reversed(st.session_state.history[-5:]):
            with st.expander(f"Topic: {item['topic'][:30]}..."):
                st.write(item['caption'])

if __name__ == "__main__":
    main()