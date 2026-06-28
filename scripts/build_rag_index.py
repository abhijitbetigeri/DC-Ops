"""
Build FAISS vector index from DC component knowledge base using MobileCLIP embeddings.

Creates a searchable index where:
- Each knowledge base entry is embedded as text using MobileCLIP's text encoder
- At inference time, YOLO crops are embedded with MobileCLIP's image encoder
- Nearest-neighbor search retrieves relevant specs/troubleshooting info

Usage:
    python scripts/build_rag_index.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch


def main():
    kb_path = Path("data/knowledge_base/dc_components.json")
    index_dir = Path("data/knowledge_base")

    with open(kb_path) as f:
        components = json.load(f)

    # Build text chunks for embedding
    chunks = []
    metadata = []
    for comp in components:
        # Create multiple chunks per component for granular retrieval
        texts = [
            f"{comp['class_name']}: {comp['description']}",
            f"{comp['class_name']} specifications: {comp['specs']}",
            f"{comp['class_name']} troubleshooting: {comp['troubleshooting']}",
            f"{comp['class_name']} LED status: {comp['led_status']}",
            f"{comp['class_name']} maintenance: {comp['maintenance']}",
        ]
        for i, text in enumerate(texts):
            chunks.append(text)
            metadata.append({
                "class_id": comp["class_id"],
                "class_name": comp["class_name"],
                "chunk_type": ["description", "specs", "troubleshooting", "led_status", "maintenance"][i],
                "text": text,
            })

    print(f"Built {len(chunks)} text chunks from {len(components)} components")

    # Load MobileCLIP for text embedding
    print("Loading MobileCLIP-S0...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP-S1", pretrained="datacompdr"
    )
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")
    model.eval()

    # Embed all text chunks
    print("Embedding text chunks...")
    all_embeddings = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i:i + batch_size]
            tokens = tokenizer(batch_texts)
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity on normalized vectors)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save index and metadata
    faiss.write_index(index, str(index_dir / "dc_rag.index"))
    with open(index_dir / "dc_rag_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save model info for loading at inference time
    info = {
        "model_name": "MobileCLIP-S1",
        "pretrained": "datacompdr",
        "embedding_dim": dim,
        "num_chunks": len(chunks),
        "num_components": len(components),
    }
    with open(index_dir / "dc_rag_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved to {index_dir}/:")
    print(f"  dc_rag.index ({(index_dir / 'dc_rag.index').stat().st_size / 1024:.0f} KB)")
    print(f"  dc_rag_metadata.json")
    print(f"  dc_rag_info.json")

    # Test retrieval
    print("\n--- Test retrieval ---")
    test_queries = [
        "server blinking amber light",
        "cable disconnected from port",
        "NVMe drive failure",
        "coolant leak detected",
        "GPU overheating compute tray",
    ]

    for query in test_queries:
        tokens = tokenizer([query])
        with torch.no_grad():
            query_emb = model.encode_text(tokens)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

        scores, indices = index.search(query_emb.cpu().numpy().astype(np.float32), k=3)
        print(f"\nQuery: '{query}'")
        for j, (score, idx) in enumerate(zip(scores[0], indices[0])):
            m = metadata[idx]
            print(f"  [{j+1}] {m['class_name']}/{m['chunk_type']} (score: {score:.3f})")
            print(f"      {m['text'][:100]}...")


if __name__ == "__main__":
    main()
