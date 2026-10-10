import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from app.config import settings

class KnowledgeRetriever:
    def __init__(self, index_path="data/knowledge/faiss_index.bin", texts_path="data/knowledge/texts.pkl"):
        self.model = SentenceTransformer(settings.embedding_model)
        self.index = None
        self.texts = []
        
        if os.path.exists(index_path) and os.path.exists(texts_path):
            self.index = faiss.read_index(index_path)
            with open(texts_path, 'rb') as f:
                self.texts = pickle.load(f)
        else:
            print("Warning: FAISS index not found. Run indexer.py first.")

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if self.index is None or not self.texts:
            return ""
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
                
        return "\n\n".join(results)
