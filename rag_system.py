# core/rag_system.py
"""
Advanced RAG (Retrieval-Augmented Generation) System with FAISS
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json
from dataclasses import dataclass

from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntry:
    """Structure for knowledge base entries"""
    idx: int
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    embedding: np.ndarray
    timestamp: datetime
    knowledge_type: str

class FAISSIndex:
    """Wrapper for FAISS index with metadata management"""
    
    def __init__(self, dimension: int, index_type: str = "FlatIP"):
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.metadata_store = {}
        self.data_store = {}
        
    def add(self, embedding: np.ndarray, data: Dict[str, Any], metadata: Dict[str, Any]) -> int:
        """Add embedding with associated data and metadata"""
        idx = self.index.ntotal
        
        # Ensure embedding is the right shape and type
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        self.index.add(embedding.astype('float32'))
        self.data_store[idx] = data
        self.metadata_store[idx] = metadata
        
        return idx
    
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings"""
        if self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        return scores[0], indices[0]
    
    def get_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get data by index"""
        return self.data_store.get(idx)
    
    def get_metadata(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get metadata by index"""
        return self.metadata_store.get(idx)

class AdvancedRAGSystem:
    """Multi-modal RAG system with FAISS for agent knowledge retrieval"""
    
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension or config.rag.vector_dimension
        self.embedding_model = SentenceTransformer(config.rag.embedding_model)
        
        # Initialize specialized indices
        self.product_index = FAISSIndex(self.dimension, "FlatIP")
        self.market_index = FAISSIndex(self.dimension, "FlatL2")
        self.strategy_index = FAISSIndex(self.dimension, "FlatIP")
        self.communication_index = FAISSIndex(self.dimension, "FlatL2")
        
        # Knowledge type mapping
        self.indices = {
            'product': self.product_index,
            'market': self.market_index,
            'strategy': self.strategy_index,
            'communication': self.communication_index
        }
        
        logger.info(f"RAG system initialized with dimension {self.dimension}")
    
    def _create_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Create embedding for text"""
        try:
            embedding = self.embedding_model.encode([text], normalize_embeddings=normalize)[0]
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self.dimension, dtype='float32')
    
    def add_product_knowledge(self, product_data: Dict[str, Any], embedding_text: str) -> int:
        """Add product knowledge to RAG system"""
        try:
            embedding = self._create_embedding(embedding_text, normalize=True)
            
            metadata = {
                'product_id': product_data.get('product_id'),
                'category': product_data.get('category'),
                'timestamp': datetime.now().isoformat(),
                'knowledge_type': 'product'
            }
            
            idx = self.product_index.add(embedding, product_data, metadata)
            logger.debug(f"Added product knowledge for {product_data.get('product_id')}")
            return idx
            
        except Exception as e:
            logger.error(f"Error adding product knowledge: {e}")
            return -1
    
    def add_market_intelligence(self, market_data: Dict[str, Any], analysis_text: str) -> int:
        """Add market intelligence to RAG system"""
        try:
            embedding = self._create_embedding(analysis_text, normalize=False)
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'market_condition': market_data.get('condition'),
                'trend': market_data.get('trend'),
                'knowledge_type': 'market'
            }
            
            idx = self.market_index.add(embedding, market_data, metadata)
            logger.debug(f"Added market intelligence: {market_data.get('condition', 'unknown')}")
            return idx
            
        except Exception as e:
            logger.error(f"Error adding market intelligence: {e}")
            return -1
    
    def add_strategy_knowledge(self, strategy_data: Dict[str, Any], strategy_text: str) -> int:
        """Add strategic knowledge to RAG system"""
        try:
            embedding = self._create_embedding(strategy_text, normalize=True)
            
            metadata = {
                'strategy_type': strategy_data.get('type'),
                'success_rate': strategy_data.get('success_rate'),
                'timestamp': datetime.now().isoformat(),
                'knowledge_type': 'strategy'
            }
            
            idx = self.strategy_index.add(embedding, strategy_data, metadata)
            logger.debug(f"Added strategy knowledge: {strategy_data.get('type', 'unknown')}")
            return idx
            
        except Exception as e:
            logger.error(f"Error adding strategy knowledge: {e}")
            return -1
    
    def add_communication_history(self, communication_data: Dict[str, Any], message_text: str) -> int:
        """Add communication to RAG system"""
        try:
            embedding = self._create_embedding(message_text, normalize=False)
            
            metadata = {
                'sender': communication_data.get('sender_id'),
                'receiver': communication_data.get('receiver_id'),
                'message_type': communication_data.get('message_type'),
                'timestamp': datetime.now().isoformat(),
                'knowledge_type': 'communication'
            }
            
            idx = self.communication_index.add(embedding, communication_data, metadata)
            logger.debug(f"Added communication: {communication_data.get('message_type', 'unknown')}")
            return idx
            
        except Exception as e:
            logger.error(f"Error adding communication: {e}")
            return -1
    
    def retrieve_relevant_knowledge(self, query: str, knowledge_type: str = "all", 
                                  k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge based on query"""
        k = k or config.rag.max_retrievals
        threshold = threshold or config.rag.similarity_threshold
        
        query_embedding = self._create_embedding(query, normalize=True)
        results = []
        
        # Determine which indices to search
        indices_to_search = []
        if knowledge_type == "all":
            indices_to_search = list(self.indices.items())
        elif knowledge_type in self.indices:
            indices_to_search = [(knowledge_type, self.indices[knowledge_type])]
        else:
            logger.warning(f"Unknown knowledge type: {knowledge_type}")
            return []
        
        # Search each relevant index
        for index_name, index in indices_to_search:
            try:
                scores, indices = index.search(query_embedding, k)
                
                for score, idx in zip(scores, indices):
                    if score >= threshold and idx in index.data_store:
                        result = {
                            'type': index_name,
                            'data': index.get_data(idx),
                            'metadata': index.get_metadata(idx),
                            'relevance_score': float(score),
                            'index': int(idx)
                        }
                        results.append(result)
                        
            except Exception as e:
                logger.error(f"Error searching {index_name} index: {e}")
                continue
        
        # Sort by relevance and return top k results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:k]
    
    def get_contextual_prompt(self, query: str, knowledge_type: str = "all", 
                            max_context_length: int = 2000) -> str:
        """Generate contextual prompt with retrieved knowledge"""
        relevant_knowledge = self.retrieve_relevant_knowledge(query, knowledge_type)
        
        if not relevant_knowledge:
            return f"""
CONTEXT: No relevant context found in knowledge base.

QUERY: {query}

Please provide a response based on your general knowledge and analysis.
"""
        
        context_parts = []
        context_length = 0
        
        for item in relevant_knowledge:
            # Create context string for this item
            if item['type'] == 'product':
                context_str = f"Product Info: {json.dumps(item['data'], indent=2)}"
            elif item['type'] == 'market':
                context_str = f"Market Intelligence: {json.dumps(item['data'], indent=2)}"
            elif item['type'] == 'strategy':
                context_str = f"Strategy Knowledge: {json.dumps(item['data'], indent=2)}"
            elif item['type'] == 'communication':
                context_str = f"Communication History: {json.dumps(item['data'], indent=2)}"
            else:
                context_str = f"Knowledge: {json.dumps(item['data'], indent=2)}"
            
            # Check if adding this context exceeds length limit
            if context_length + len(context_str) > max_context_length:
                break
            
            context_parts.append(context_str)
            context_length += len(context_str)
        
        context = "\n\n".join(context_parts)
        
        return f"""
CONTEXT FROM KNOWLEDGE BASE:
{context}

QUERY: {query}

Please provide a response based on the above context and your analysis.
"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            'total_entries': 0,
            'entries_by_type': {},
            'embedding_dimension': self.dimension,
            'model_name': config.rag.embedding_model
        }
        
        for name, index in self.indices.items():
            count = index.index.ntotal
            stats['entries_by_type'][name] = count
            stats['total_entries'] += count
        
        return stats
    
    def clear_knowledge_base(self, knowledge_type: Optional[str] = None):
        """Clear knowledge base entries"""
        if knowledge_type and knowledge_type in self.indices:
            # Clear specific index
            index = self.indices[knowledge_type]
            index.index.reset()
            index.data_store.clear()
            index.metadata_store.clear()
            logger.info(f"Cleared {knowledge_type} knowledge base")
        elif knowledge_type is None:
            # Clear all indices
            for name, index in self.indices.items():
                index.index.reset()
                index.data_store.clear()
                index.metadata_store.clear()
            logger.info("Cleared all knowledge bases")
        else:
            logger.warning(f"Unknown knowledge type: {knowledge_type}")
