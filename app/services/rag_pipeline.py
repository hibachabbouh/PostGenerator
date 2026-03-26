import os

from dotenv import load_dotenv
from groq import Groq

from ml.retriever import SmartRetriever

retriever = None
client = None


def _get_retriever() -> SmartRetriever:
    global retriever
    if retriever is None:
        retriever = SmartRetriever()
    return retriever


def _get_client() -> Groq:
    global client
    if client is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Set it in environment or in a .env file at project root."
            )
        client = Groq(api_key=api_key)
    return client

def generate_caption(topic, style):
    examples = _get_retriever().search(topic, style=style, k=5)

    examples_text = "\n".join([
        f"- {e['clean_text']} (tone: {e['tone']}, hook: {e['hook_strength']})"
        for e in examples
    ])

    prompt = f"""
You are an Instagram expert.

Topic: {topic}
Style: {style}

Here are high-quality examples:
{examples_text}

Generate a NEW caption that:
- Matches the style
- Has strong hook
- Uses emojis
- Includes hashtags

Return only the caption.
"""

    response = _get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content