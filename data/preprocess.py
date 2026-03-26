
import json
import os
import re
import time
from typing import Any
import emoji

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"
INPUT_CSV = "data/raw/social_media_captions.csv"
OUTPUT_CSV = "data/processed/llm_enriched.csv"
BATCH_SIZE = 10
SLEEP_SECONDS = 1
MAX_ROWS = 100

VALID_STYLES = {"funny", "motivational", "aesthetic", "general"}


def extract_caption(raw_text: str) -> str:
    """Extract caption from raw text format."""
    text = str(raw_text)
    if "### Assistant:" in text:
        return text.split("### Assistant:", maxsplit=1)[1].strip()
    return text.strip()


def extract_emojis(text: str) -> list[str]:
    """Extract all emojis from text."""
    return [char for char in text if char in emoji.EMOJI_DATA]


def analyze_caption_structure(text: str) -> dict[str, Any]:
    """Extract structural features without destroying the original text."""
    hashtags = re.findall(r'#\w+', text)
    mentions = re.findall(r'@\w+', text)
    urls = re.findall(r'http\S+', text)
    emojis = extract_emojis(text)
    
    clean_text = re.sub(r'http\S+', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    first_line = lines[0] if lines else clean_text[:100]

    has_question = '?' in text
    has_exclamation = '!' in text
    has_ellipsis = '...' in text or '…' in text
    text_without_tags = re.sub(r'#\w+|@\w+', '', clean_text)
    word_count = len(text_without_tags.split())
    
    return {
        'original_text': text,
        'clean_text': clean_text,
        'hashtags': hashtags,
        'mentions': mentions,
        'emojis': emojis,
        'emoji_count': len(emojis),
        'has_url': len(urls) > 0,
        'has_question': has_question,
        'has_exclamation': has_exclamation,
        'has_ellipsis': has_ellipsis,
        'word_count': word_count,
        'char_count': len(clean_text),
        'line_count': len(lines),
        'first_line': first_line,
        'hashtag_count': len(hashtags),
        'mention_count': len(mentions),
    }


def prepare_inputs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Prepare and deduplicate input data with rich feature extraction."""
    prepared_data: list[dict[str, Any]] = []

    for raw in df["text"].dropna().astype(str).tolist():
        caption = extract_caption(raw)
        if len(caption.strip()) <= 5:
            continue
            
        features = analyze_caption_structure(caption)
        if features['word_count'] > 0:
            prepared_data.append(features)
    unique_by_text: dict[str, dict[str, Any]] = {}
    for item in prepared_data:
        unique_by_text.setdefault(item["clean_text"], item)

    return list(unique_by_text.values())


def build_batch_prompt(batch_data: list[dict[str, Any]]) -> str:
    """Build prompt for LLM to classify style and extract semantic features."""
    
    captions_text = "\n\n".join(
        f"{idx + 1}. {item['clean_text'][:200]}" 
        for idx, item in enumerate(batch_data)
    )
    
    return f"""You are an Instagram content analyzer. For each caption, classify its style and extract key features.

Return ONLY a JSON array with this exact structure (no markdown, no explanation):
[
    {{
        "style": "funny|motivational|aesthetic|general",
        "primary_theme": "brief theme description",
        "tone": "casual|professional|playful|inspirational|educational",
        "has_cta": true/false,
        "hook_strength": 1-5,
        "target_audience": "who this appeals to"
    }}
]

STYLE DEFINITIONS:
- funny: Humorous, witty, comedic, memes, jokes
- motivational: Inspirational, encouraging, goal-oriented, success-focused
- aesthetic: Visual-focused, artsy, mood-based, vibe-centric, minimalist
- general: News, updates, informational, neutral content

Rules:
- Return EXACTLY {len(batch_data)} objects in the array
- Each field is required
- hook_strength: 1=weak, 5=very compelling first line
- has_cta: true if asking for action (comment, share, click, buy, follow, etc.)

Captions:
{captions_text}
""".strip()


def enrich_with_llm(
    client: Groq, 
    batch_data: list[dict[str, Any]], 
    model_name: str
) -> list[dict[str, Any]]:
    """Use LLM to classify style and extract semantic features."""
    
    prompt = build_batch_prompt(batch_data)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    output = response.choices[0].message.content
    
    # Clean potential markdown wrapping
    output = output.strip()
    if output.startswith("```json"):
        output = output[7:]
    if output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()
    
    parsed = json.loads(output)

    if not isinstance(parsed, list):
        raise ValueError("LLM response is not a JSON list.")
    
    if len(parsed) != len(batch_data):
        raise ValueError(
            f"LLM returned {len(parsed)} items but expected {len(batch_data)}"
        )

    return parsed


def run() -> None:
    """Main preprocessing pipeline."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable.")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    # Load and validate input
    df = pd.read_csv(INPUT_CSV)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # Prepare inputs with feature extraction
    print("Extracting features from captions...")
    prepared = prepare_inputs(df)
    
    if MAX_ROWS > 0:
        prepared = prepared[:MAX_ROWS]

    print(f"Prepared {len(prepared)} unique captions")

    # Process in batches with LLM
    enriched_data: list[dict[str, Any]] = []
    client = Groq(api_key=api_key)
    print(f"Using model: {MODEL_NAME}")
    print(f"Processing {len(prepared)} captions in batches of {BATCH_SIZE}")

    for start in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[start:start + BATCH_SIZE]
        
        try:
            llm_results = enrich_with_llm(client, batch, MODEL_NAME)
        except Exception as error:
            print(f"⚠️  Batch starting at index {start} failed: {error}")
            # Fallback: use default values
            llm_results = [
                {
                    "style": "general",
                    "primary_theme": "unknown",
                    "tone": "casual",
                    "has_cta": False,
                    "hook_strength": 3,
                    "target_audience": "general"
                }
                for _ in batch
            ]

        # Combine structural features with LLM analysis
        for source, llm_data in zip(batch, llm_results):
            # Validate style
            style = llm_data.get("style", "general")
            if style not in VALID_STYLES:
                style = "general"
            
            # Combine all features
            enriched_data.append({
                # Original content
                "original_text": source["original_text"],
                "clean_text": source["clean_text"],
                
                # User-selectable style (static list)
                "style": style,
                
                # Structural features
                "hashtags": " ".join(source["hashtags"]),
                "hashtag_count": source["hashtag_count"],
                "mentions": " ".join(source["mentions"]),
                "mention_count": source["mention_count"],
                "emojis": "".join(source["emojis"]),
                "emoji_count": source["emoji_count"],
                
                # Text features
                "word_count": source["word_count"],
                "char_count": source["char_count"],
                "line_count": source["line_count"],
                "first_line": source["first_line"],
                
                # Pattern detection
                "has_question": source["has_question"],
                "has_exclamation": source["has_exclamation"],
                "has_url": source["has_url"],
                
                # LLM semantic analysis
                "primary_theme": llm_data.get("primary_theme", "unknown"),
                "tone": llm_data.get("tone", "casual"),
                "has_cta": llm_data.get("has_cta", False),
                "hook_strength": llm_data.get("hook_strength", 3),
                "target_audience": llm_data.get("target_audience", "general"),
            })

        processed_count = min(start + BATCH_SIZE, len(prepared))
        print(f"✓ Processed {processed_count}/{len(prepared)}")
        time.sleep(SLEEP_SECONDS)

    # Save enriched dataset
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    output_df = pd.DataFrame(enriched_data)
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    


if __name__ == "__main__":
    run()