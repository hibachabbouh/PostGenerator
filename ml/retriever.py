import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Literal
import random


class SmartRetriever:
    """
    Advanced retrieval system for Instagram caption generation.
    Supports multiple retrieval strategies and hybrid ranking.
    """
    
    def __init__(self, embeddings_dir: str = "embeddings"):
        """Initialize retriever with pre-computed embeddings and indices."""
        
        # Load main index (all posts)
        self.main_index = faiss.read_index(f"{embeddings_dir}/faiss_index.bin")
        
        # Load metadata
        with open(f"{embeddings_dir}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load style-specific indices
        with open(f"{embeddings_dir}/style_indices.pkl", "rb") as f:
            self.style_indices = pickle.load(f)
        
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"✓ Loaded {len(self.metadata)} posts")
        print(f"✓ Available styles: {list(self.style_indices.keys())}")
    
    def _create_query_embedding(self, query: str, style: Optional[str] = None) -> np.ndarray:
        """Create query embedding with optional style context."""
        
        # Add style context to query for better matching
        if style:
            enriched_query = f"{query} | Style: {style}"
        else:
            enriched_query = query
        
        embedding = self.model.encode([enriched_query])[0]
        return embedding

    def search(self, *args, **kwargs) -> List[Dict]:
        """Backward-compatible alias for semantic search."""
        return self.search_semantic(*args, **kwargs)
    
    def _hybrid_score(
        self,
        semantic_distance: float,
        post: Dict,
        target_length: Optional[int] = None,
        prefer_high_hooks: bool = False,
        prefer_with_cta: bool = False
    ) -> float:
        """
        Calculate hybrid score combining semantic similarity with other factors.
        Lower is better (since we're using L2 distance).
        """
        
        # Start with semantic distance (lower = more similar)
        score = semantic_distance
        
        # Penalize length mismatch if target specified
        if target_length is not None:
            length_diff = abs(post.get('word_count', 0) - target_length)
            # Normalize: 10 word difference = 0.1 penalty
            score += length_diff * 0.01
        
        # Boost posts with high hook strength
        if prefer_high_hooks:
            hook_strength = post.get('hook_strength', 3)
            # Higher hooks get lower scores (better ranking)
            score -= (hook_strength - 3) * 0.05
        
        # Boost posts with CTAs
        if prefer_with_cta and post.get('has_cta', False):
            score -= 0.1  # Small boost
        
        return score
    
    def search_semantic(
        self,
        query: str,
        style: Optional[str] = None,
        k: int = 5,
        target_length: Optional[int] = None,
        prefer_high_hooks: bool = False,
        prefer_with_cta: bool = False
    ) -> List[Dict]:
        """
        Semantic similarity search with hybrid ranking.
        
        Args:
            query: User's prompt or theme
            style: Filter by style (funny/motivational/aesthetic/general)
            k: Number of results to return
            target_length: Preferred word count (will prioritize similar lengths)
            prefer_high_hooks: Boost posts with high hook_strength
            prefer_with_cta: Boost posts with call-to-action
        
        Returns:
            List of post dictionaries, ranked by hybrid score
        """
        
        # Create query embedding
        query_embedding = self._create_query_embedding(query, style)
        
        # Search appropriate index
        if style and style in self.style_indices:
            # Use style-specific index for efficiency
            index_data = self.style_indices[style]
            index = index_data['index']
            original_positions = index_data['original_positions']
            
            # Search (get more candidates for re-ranking)
            D, I = index.search(
                np.array([query_embedding], dtype='float32'),
                min(k * 3, len(original_positions))
            )
            
            # Map back to original metadata positions
            results = [self.metadata[original_positions[i]] for i in I[0]]
            distances = D[0]
            
        else:
            # Use main index (all posts)
            D, I = self.main_index.search(
                np.array([query_embedding], dtype='float32'),
                k * 3  # Get more candidates for filtering/re-ranking
            )
            
            results = [self.metadata[i] for i in I[0]]
            distances = D[0]
            
            # Filter by style if specified
            if style:
                filtered = []
                filtered_distances = []
                for post, dist in zip(results, distances):
                    if post.get('style') == style:
                        filtered.append(post)
                        filtered_distances.append(dist)
                results = filtered
                distances = np.array(filtered_distances)
        
        # Apply hybrid scoring
        scored_results = []
        for post, dist in zip(results, distances):
            hybrid_score = self._hybrid_score(
                dist,
                post,
                target_length=target_length,
                prefer_high_hooks=prefer_high_hooks,
                prefer_with_cta=prefer_with_cta
            )
            scored_results.append((hybrid_score, post))
        
        # Sort by hybrid score and return top k
        scored_results.sort(key=lambda x: x[0])
        return [post for _, post in scored_results[:k]]
    
    def search_random(
        self,
        style: str,
        k: int = 5,
        min_hook_strength: Optional[int] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None
    ) -> List[Dict]:
        """
        Random sampling within a style (good for diversity).
        
        Args:
            style: Required style filter
            k: Number of results
            min_hook_strength: Only include posts with hook >= this value
            min_word_count: Only include posts with word_count >= this
            max_word_count: Only include posts with word_count <= this
        
        Returns:
            Random sample of posts matching criteria
        """
        
        # Filter by style
        candidates = [p for p in self.metadata if p.get('style') == style]
        
        # Apply additional filters
        if min_hook_strength is not None:
            candidates = [p for p in candidates if p.get('hook_strength', 0) >= min_hook_strength]
        
        if min_word_count is not None:
            candidates = [p for p in candidates if p.get('word_count', 0) >= min_word_count]
        
        if max_word_count is not None:
            candidates = [p for p in candidates if p.get('word_count', 999) <= max_word_count]
        
        # Sample
        sample_size = min(k, len(candidates))
        return random.sample(candidates, sample_size)
    
    def search_best_hooks(
        self,
        style: str,
        k: int = 5,
        min_hook_strength: int = 4
    ) -> List[Dict]:
        """
        Get posts with the strongest hooks in a given style.
        Perfect for teaching the generator how to write compelling openings.
        
        Args:
            style: Required style filter
            k: Number of results
            min_hook_strength: Minimum hook strength (default 4)
        
        Returns:
            Posts sorted by hook_strength descending
        """
        
        # Filter by style and hook strength
        candidates = [
            p for p in self.metadata 
            if p.get('style') == style and p.get('hook_strength', 0) >= min_hook_strength
        ]
        
        # Sort by hook strength (descending)
        candidates.sort(key=lambda x: x.get('hook_strength', 0), reverse=True)
        
        return candidates[:k]
    
    def search_by_filters(
        self,
        style: Optional[str] = None,
        tone: Optional[str] = None,
        has_cta: Optional[bool] = None,
        min_emoji_count: Optional[int] = None,
        max_emoji_count: Optional[int] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None,
        k: int = 10
    ) -> List[Dict]:
        """
        Advanced filtering for very specific retrieval needs.
        
        Example: Get short, funny posts with CTAs and lots of emojis
        → style="funny", has_cta=True, max_word_count=15, min_emoji_count=3
        """
        
        candidates = self.metadata.copy()
        
        # Apply each filter
        if style:
            candidates = [p for p in candidates if p.get('style') == style]
        
        if tone:
            candidates = [p for p in candidates if p.get('tone') == tone]
        
        if has_cta is not None:
            candidates = [p for p in candidates if p.get('has_cta') == has_cta]
        
        if min_emoji_count is not None:
            candidates = [p for p in candidates if p.get('emoji_count', 0) >= min_emoji_count]
        
        if max_emoji_count is not None:
            candidates = [p for p in candidates if p.get('emoji_count', 999) <= max_emoji_count]
        
        if min_word_count is not None:
            candidates = [p for p in candidates if p.get('word_count', 0) >= min_word_count]
        
        if max_word_count is not None:
            candidates = [p for p in candidates if p.get('word_count', 999) <= max_word_count]
        
        return candidates[:k]
    
    def get_style_statistics(self, style: str) -> Dict:
        """
        Get statistics for a specific style to inform generation.
        
        Returns:
            Dictionary with averages, distributions, common patterns
        """
        
        posts = [p for p in self.metadata if p.get('style') == style]
        
        if not posts:
            return {"error": f"No posts found for style '{style}'"}
        
        # Calculate statistics
        word_counts = [p.get('word_count', 0) for p in posts]
        emoji_counts = [p.get('emoji_count', 0) for p in posts]
        hook_strengths = [p.get('hook_strength', 0) for p in posts]
        
        # Count patterns
        has_cta_count = sum(1 for p in posts if p.get('has_cta', False))
        has_question_count = sum(1 for p in posts if p.get('has_question', False))
        
        # Tone distribution
        tones = {}
        for p in posts:
            tone = p.get('tone', 'unknown')
            tones[tone] = tones.get(tone, 0) + 1
        
        return {
            'total_posts': len(posts),
            'avg_word_count': np.mean(word_counts),
            'avg_emoji_count': np.mean(emoji_counts),
            'avg_hook_strength': np.mean(hook_strengths),
            'has_cta_pct': (has_cta_count / len(posts)) * 100,
            'has_question_pct': (has_question_count / len(posts)) * 100,
            'tone_distribution': tones,
            'word_count_range': (min(word_counts), max(word_counts))
        }



    
 