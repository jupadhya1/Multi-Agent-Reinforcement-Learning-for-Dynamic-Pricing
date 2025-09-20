# config/settings.py
"""
Configuration settings for the LLM Multi-Agent Pricing System
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    """LLM-specific configuration"""
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    model_name: str = "gpt-4"
    fallback_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1500
    timeout: int = 30
    retry_attempts: int = 3

@dataclass
class AgentConfig:
    """Agent behavior configuration"""
    temperature_settings: Dict[str, float] = None
    personality_traits: Dict[str, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.temperature_settings is None:
            self.temperature_settings = {
                "aggressive": 0.8,
                "conservative": 0.3,
                "adaptive": 0.6,
                "collaborative": 0.5
            }
        
        if self.personality_traits is None:
            self.personality_traits = {
                "aggressive": {"risk_tolerance": 0.9, "market_focus": 0.8},
                "conservative": {"risk_tolerance": 0.2, "stability_focus": 0.9},
                "adaptive": {"learning_rate": 0.7, "flexibility": 0.8},
                "collaborative": {"cooperation": 0.8, "ecosystem_focus": 0.7}
            }

@dataclass
class RAGConfig:
    """RAG system configuration"""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_dimension: int = 384
    faiss_index_type: str = "FlatIP"
    similarity_threshold: float = 0.7
    max_retrievals: int = 5

@dataclass
class MemoryConfig:
    """Memory system configuration"""
    use_mem0: bool = True
    local_memory_size: int = 10000
    semantic_similarity_threshold: float = 0.8
    memory_retention_days: int = 30

@dataclass
class SimulationConfig:
    """Simulation parameters"""
    max_weeks: int = 52
    products_per_agent: int = 5
    total_products: int = 100
    market_categories: list = None
    
    def __post_init__(self):
        if self.market_categories is None:
            self.market_categories = [
                'Electronics', 'Clothing', 'Home & Garden', 'Sports', 
                'Books', 'Toys', 'Health', 'Automotive'
            ]

@dataclass
class MarketConfig:
    """Market environment configuration"""
    base_competition_level: float = 0.5
    base_demand_volatility: float = 0.3
    seasonal_factor_range: tuple = (0.8, 1.3)
    price_elasticity_range: tuple = (-1.5, -0.3)
    economic_indicator_range: tuple = (0.2, 0.9)

class SystemConfig:
    """Main system configuration"""
    
    def __init__(self):
        self.llm = LLMConfig()
        self.agents = AgentConfig()
        self.rag = RAGConfig()
        self.memory = MemoryConfig()
        self.simulation = SimulationConfig()
        self.market = MarketConfig()
        
        # Validate critical settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.llm.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.simulation.products_per_agent <= 0:
            raise ValueError("Products per agent must be positive")
        
        if self.rag.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'llm': self.llm.__dict__,
            'agents': self.agents.__dict__,
            'rag': self.rag.__dict__,
            'memory': self.memory.__dict__,
            'simulation': self.simulation.__dict__,
            'market': self.market.__dict__
        }

# Global configuration instance
config = SystemConfig()
