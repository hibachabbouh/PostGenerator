import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from typing import Dict, List
INPUT_CSV = "data/processed/llm_enriched.csv"
EMBEDDINGS_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")
STYLE_INDICES_PATH = os.path.join(EMBEDDINGS_DIR, "style_indices.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"


def create_rich_embedding_text(row: Dict) -> str:
    """
    Create enriched text for embedding that includes semantic context.
    This helps the model understand not just WHAT the caption says,
    but HOW it's structured and WHO it's for.
    """
    parts = []
    
    # Main caption text
    parts.append(row["clean_text"])
    
    # Add style and tone as context (helps cluster similar vibes)
    parts.append(f"Style: {row['style']}")
    parts.append(f"Tone: {row['tone']}")
    
    # Add theme (helps match similar topics)
    parts.append(f"Theme: {row['primary_theme']}")
    
    # Add audience context (helps match target demographic)
    parts.append(f"Audience: {row['target_audience']}")
    
    # Add hashtags (strong signal for topic/niche)
    if row.get("hashtags") and str(row["hashtags"]).strip():
        parts.append(f"Tags: {row['hashtags']}")
    
    # Combine with separators
    return " | ".join(parts)


def create_style_specific_indices(
    embeddings: np.ndarray, 
    df: pd.DataFrame
) -> Dict[str, faiss.Index]:
    """
    Create separate FAISS indices for each style.
    This makes style-filtered search much faster and more accurate.
    """
    style_indices = {}
    
    for style in df["style"].unique():
        # Get indices of all posts with this style
        mask = df["style"] == style
        style_positions = np.where(mask)[0]
        
        # Extract embeddings for this style
        style_embeddings = embeddings[style_positions]
        
        # Create FAISS index for this style
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(style_embeddings.astype('float32'))
        
        # Store index and mapping to original positions
        style_indices[style] = {
            'index': index,
            'original_positions': style_positions.tolist()
        }
        
        print(f"  ✓ {style}: {len(style_positions)} posts indexed")
    
    return style_indices


def run() -> None:
    """Main embedding creation pipeline."""
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} captions")
    
    # Validate required columns
    required_cols = ["clean_text", "style", "tone", "primary_theme", "target_audience"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Create rich embedding texts
    print("\nCreating enriched embedding texts...")
    embedding_texts = [create_rich_embedding_text(row) for _, row in df.iterrows()]
    
    # Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        embedding_texts, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Create main FAISS index (for cross-style semantic search)
    print("\nCreating main FAISS index...")
    dimension = embeddings.shape[1]
    main_index = faiss.IndexFlatL2(dimension)
    main_index.add(embeddings.astype('float32'))
    print(f"Main index created with {main_index.ntotal} vectors")
    
    # Create style-specific indices (for efficient style-filtered search)
    print("\nCreating style-specific indices...")
    style_indices = create_style_specific_indices(embeddings, df)
    
    # Prepare metadata
    print("\nPreparing metadata...")
    metadata = df.to_dict("records")
    
    # Add some computed fields for easier filtering later
    for i, row in enumerate(metadata):
        row['_index'] = i  # Original position
        row['_embedding_text'] = embedding_texts[i]  # What was embedded
    
    # Create output directory
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    # Save everything
    print("\nSaving artifacts...")
    
    # Save main index
    faiss.write_index(main_index, FAISS_INDEX_PATH)
    print(f"Main index saved to {FAISS_INDEX_PATH}")
    
    # Save metadata
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f" Metadata saved to {METADATA_PATH}")
    
    # Save style indices
    with open(STYLE_INDICES_PATH, "wb") as f:
        pickle.dump(style_indices, f)
    print(f"Style indices saved to {STYLE_INDICES_PATH}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EMBEDDING CREATION COMPLETE")
    print("="*60)
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Model: {MODEL_NAME}")
    print("\nStyle breakdown:")
    for style, data in style_indices.items():
        print(f"  {style}: {len(data['original_positions'])} posts")
    
    print("\nFiles created:")
    print(f"  - {FAISS_INDEX_PATH}")
    print(f"  - {METADATA_PATH}")
    print(f"  - {STYLE_INDICES_PATH}")
    
    # Quality check: sample similar posts
    print("\n" + "="*60)
    print("QUALITY CHECK: Sample similarity search")
    print("="*60)
    
    test_query = df.iloc[0]["clean_text"]
    test_embedding = model.encode([test_query])[0:1]
    D, I = main_index.search(test_embedding.astype('float32'), 3)
    
    print(f"\nQuery: {test_query[:80]}...")
    print(f"Style: {df.iloc[0]['style']}")
    print("\nTop 3 similar posts:")
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        similar_post = df.iloc[idx]
        print(f"\n{rank}. Distance: {dist:.4f}")
        print(f"   Style: {similar_post['style']}")
        print(f"   Text: {similar_post['clean_text'][:80]}...")


if __name__ == "__main__":
    run()