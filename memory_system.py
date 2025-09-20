# core/memory_system.py
"""
Advanced Memory System with Mem0 integration and structured storage
"""

import asyncio
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from sentence_transformers import SentenceTransformer

from models.data_models import PricingDecision, MarketAnalysis, PerformanceMetrics
from config.settings import config

logger = logging.getLogger(__name__)

# Try to import Mem0, fall back to local memory if unavailable
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("Mem0 not available, using local memory only")

class LocalMemoryStore:
    """Local memory store as fallback when Mem0 is unavailable"""
    
    def __init__(self, max_size: int = 10000):
        self.memories = {}
        self.max_size = max_size
        self.creation_times = {}
    
    async def add(self, text: str, user_id: str) -> str:
        """Add memory entry"""
        memory_id = f"{user_id}_{len(self.memories.get(user_id, []))}"
        
        if user_id not in self.memories:
            self.memories[user_id] = []
        
        # Check size limit
        if len(self.memories[user_id]) >= self.max_size:
            # Remove oldest memory
            oldest_id = min(self.creation_times.keys(), key=self.creation_times.get)
            self.memories[user_id].remove(oldest_id)
            del self.creation_times[oldest_id]
        
        self.memories[user_id].append({
            'id': memory_id,
            'text': text,
            'timestamp': datetime.now()
        })
        self.creation_times[memory_id] = datetime.now()
        
        return memory_id
    
    async def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """Search memories using simple text matching"""
        if user_id not in self.memories:
            return []
        
        # Simple keyword-based search
        query_words = query.lower().split()
        results = []
        
        for memory in self.memories[user_id]:
            memory_text = memory['text'].lower()
            score = sum(1 for word in query_words if word in memory_text)
            
            if score > 0:
                results.append({
                    'text': memory['text'],
                    'score': score / len(query_words),
                    'timestamp': memory['timestamp']
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

class AdvancedMemorySystem:
    """Enhanced memory system with Mem0 integration and structured storage"""
    
    def __init__(self):
        self.use_mem0 = config.memory.use_mem0 and MEM0_AVAILABLE
        
        # Initialize Mem0 if available
        if self.use_mem0:
            try:
                self.mem0_client = Memory()
                logger.info("Mem0 client initialized successfully")
            except Exception as e:
                logger.error(f"Mem0 initialization failed: {e}")
                self.use_mem0 = False
        
        # Local memory structures
        self.agent_memories = defaultdict(list)
        self.market_memories = []
        self.strategy_memories = defaultdict(list)
        self.communication_memories = []
        self.performance_memories = defaultdict(list)
        
        # Local memory store as fallback
        self.local_store = LocalMemoryStore(config.memory.local_memory_size)
        
        # Memory indices for fast retrieval
        self.memory_embeddings = {}
        self.embedding_model = SentenceTransformer(config.rag.embedding_model)
        
        logger.info(f"Memory system initialized (Mem0: {self.use_mem0})")
    
    async def store_agent_decision(self, agent_id: str, decision: PricingDecision, 
                                 context: Dict[str, Any]) -> bool:
        """Store agent pricing decision with full context"""
        try:
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'decision': decision.model_dump(),
                'context': context,
                'memory_type': 'pricing_decision'
            }
            
            # Store in Mem0 if available
            if self.use_mem0:
                try:
                    memory_text = (
                        f"Agent {agent_id} decided to change price for {decision.product_id} "
                        f"from ${decision.current_price:.2f} to ${decision.recommended_price:.2f}. "
                        f"Reasoning: {decision.reasoning}. "
                        f"Risk: {decision.risk_assessment}. "
                        f"Expected impact: {decision.expected_impact}. "
                        f"Context: {context}"
                    )
                    await asyncio.to_thread(self.mem0_client.add, memory_text, user_id=agent_id)
                    logger.debug(f"Stored decision in Mem0 for {agent_id}")
                except Exception as e:
                    logger.warning(f"Mem0 storage failed for {agent_id}: {e}")
                    # Fall back to local storage
                    await self.local_store.add(memory_text, agent_id)
            else:
                # Use local store
                memory_text = f"Decision: {decision.reasoning}"
                await self.local_store.add(memory_text, agent_id)
            
            # Store locally for fast access
            self.agent_memories[agent_id].append(memory_entry)
            
            # Create embedding for similarity search
            text_representation = f"{decision.reasoning} {decision.expected_impact}"
            embedding = self.embedding_model.encode([text_representation])[0]
            
            memory_key = f"{agent_id}_{len(self.agent_memories[agent_id])}"
            self.memory_embeddings[memory_key] = {
                'embedding': embedding,
                'memory': memory_entry
            }
            
            # Cleanup old memories if needed
            await self._cleanup_old_memories(agent_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing agent decision for {agent_id}: {e}")
            return False
    
    async def store_market_intelligence(self, analysis: MarketAnalysis, source_agent: str) -> bool:
        """Store market intelligence from agents"""
        try:
            memory_entry = {
                'timestamp': datetime.now().isoformat(),
                'source_agent': source_agent,
                'analysis': analysis.model_dump(),
                'memory_type': 'market_intelligence'
            }
            
            if self.use_mem0:
                try:
                    memory_text = (
                        f"Market analysis by {source_agent}: {analysis.market_trend} trend, "
                        f"competition level {analysis.competition_level:.2f}, "
                        f"strategy: {analysis.recommended_strategy}. "
                        f"Confidence: {analysis.confidence_score:.2f}. "
                        f"Insights: {', '.join(analysis.key_insights[:3])}"
                    )
                    await asyncio.to_thread(
                        self.mem0_client.add, 
                        memory_text, 
                        user_id="market_intelligence"
                    )
                    logger.debug(f"Stored market analysis in Mem0 from {source_agent}")
                except Exception as e:
                    logger.warning(f"Mem0 storage failed for market analysis: {e}")
                    await self.local_store.add(memory_text, "market_intelligence")
            
            self.market_memories.append(memory_entry)
            return True
            
        except Exception as e:
            logger.error(f"Error storing market intelligence: {e}")
            return False
    
    async def store_performance_metrics(self, agent_id: str, metrics: PerformanceMetrics, 
                                      week: int) -> bool:
        """Store agent performance metrics"""
        try:
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'week': week,
                'metrics': metrics.model_dump()
            }
            
            self.performance_memories[agent_id].append(performance_entry)
            
            # Store summary in Mem0 if available
            if self.use_mem0:
                try:
                    memory_text = (
                        f"Week {week} performance for {agent_id}: "
                        f"Revenue ${metrics.revenue:.2f}, "
                        f"Market share {metrics.market_share:.1f}%, "
                        f"Position: {metrics.competitive_position}"
                    )
                    await asyncio.to_thread(
                        self.mem0_client.add, 
                        memory_text, 
                        user_id=f"{agent_id}_performance"
                    )
                except Exception as e:
                    logger.warning(f"Mem0 performance storage failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            return False
    
    async def retrieve_similar_decisions(self, agent_id: str, current_context: str, 
                                       k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar past decisions using semantic search"""
        try:
            # Try Mem0 first if available
            if self.use_mem0:
                try:
                    results = await asyncio.to_thread(
                        self.mem0_client.search, 
                        current_context, 
                        user_id=agent_id
                    )
                    if results:
                        # Convert Mem0 results to our format
                        formatted_results = []
                        for result in results[:k]:
                            formatted_results.append({
                                'memory_type': 'pricing_decision',
                                'content': result.get('text', ''),
                                'relevance_score': result.get('score', 0.5),
                                'timestamp': result.get('timestamp', datetime.now().isoformat())
                            })
                        return formatted_results
                except Exception as e:
                    logger.warning(f"Mem0 retrieval failed for {agent_id}: {e}")
            
            # Fallback to local semantic search
            if not self.agent_memories[agent_id]:
                return []
            
            query_embedding = self.embedding_model.encode([current_context])[0]
            similarities = []
            
            for memory_key, memory_data in self.memory_embeddings.items():
                if memory_key.startswith(agent_id):
                    similarity = np.dot(query_embedding, memory_data['embedding'])
                    similarities.append({
                        'similarity': float(similarity),
                        'memory': memory_data['memory']
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return [item['memory'] for item in similarities[:k]]
            
        except Exception as e:
            logger.error(f"Error retrieving similar decisions for {agent_id}: {e}")
            return []
    
    async def get_agent_performance_trends(self, agent_id: str, 
                                         lookback_weeks: int = 10) -> Dict[str, Any]:
        """Analyze agent performance trends from memory"""
        try:
            if agent_id not in self.performance_memories:
                return {}
            
            recent_performance = self.performance_memories[agent_id][-lookback_weeks:]
            
            if len(recent_performance) < 2:
                return {}
            
            # Extract metrics
            revenues = [p['metrics']['revenue'] for p in recent_performance]
            market_shares = [p['metrics']['market_share'] for p in recent_performance]
            
            # Calculate trends
            revenue_trend = np.polyfit(range(len(revenues)), revenues, 1)[0] if len(revenues) > 1 else 0
            market_share_trend = np.polyfit(range(len(market_shares)), market_shares, 1)[0] if len(market_shares) > 1 else 0
            
            return {
                'revenue_trend': float(revenue_trend),
                'avg_market_share': float(np.mean(market_shares)),
                'market_share_trend': float(market_share_trend),
                'performance_stability': float(np.std(revenues)),
                'total_decisions': len(self.agent_memories[agent_id]),
                'weeks_analyzed': len(recent_performance),
                'latest_performance': recent_performance[-1]['metrics'] if recent_performance else {}
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends for {agent_id}: {e}")
            return {}
    
    async def get_market_intelligence_summary(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent market intelligence"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
            recent_intelligence = [
                intel for intel in self.market_memories
                if datetime.fromisoformat(intel['timestamp']) > cutoff_time
            ]
            
            if not recent_intelligence:
                return {}
            
            # Aggregate insights
            trends = [intel['analysis']['market_trend'] for intel in recent_intelligence]
            competition_levels = [intel['analysis']['competition_level'] for intel in recent_intelligence]
            confidence_scores = [intel['analysis']['confidence_score'] for intel in recent_intelligence]
            
            return {
                'total_analyses': len(recent_intelligence),
                'dominant_trend': max(set(trends), key=trends.count) if trends else 'unknown',
                'avg_competition_level': float(np.mean(competition_levels)) if competition_levels else 0.5,
                'avg_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0.5,
                'timeframe_hours': lookback_hours
            }
            
        except Exception as e:
            logger.error(f"Error getting market intelligence summary: {e}")
            return {}
    
    async def _cleanup_old_memories(self, agent_id: str):
        """Clean up old memories to prevent memory bloat"""
        try:
            max_memories = config.memory.local_memory_size
            
            if len(self.agent_memories[agent_id]) > max_memories:
                # Remove oldest memories
                excess_count = len(self.agent_memories[agent_id]) - max_memories
                self.agent_memories[agent_id] = self.agent_memories[agent_id][excess_count:]
                
                # Clean up corresponding embeddings
                old_keys = [
                    key for key in self.memory_embeddings.keys()
                    if key.startswith(agent_id) and 
                    int(key.split('_')[-1]) <= excess_count
                ]
                
                for key in old_keys:
                    del self.memory_embeddings[key]
                
                logger.debug(f"Cleaned up {excess_count} old memories for {agent_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up memories for {agent_id}: {e}")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            stats = {
                'mem0_enabled': self.use_mem0,
                'total_agent_memories': sum(len(memories) for memories in self.agent_memories.values()),
                'total_market_memories': len(self.market_memories),
                'total_performance_memories': sum(len(memories) for memories in self.performance_memories.values()),
                'total_embeddings': len(self.memory_embeddings),
                'agents_with_memories': len(self.agent_memories),
                'memory_by_agent': {
                    agent_id: len(memories) 
                    for agent_id, memories in self.agent_memories.items()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {}
    
    async def clear_memories(self, agent_id: Optional[str] = None, memory_type: Optional[str] = None):
        """Clear memories by agent or type"""
        try:
            if agent_id:
                # Clear specific agent's memories
                if memory_type == 'decisions':
                    self.agent_memories[agent_id].clear()
                elif memory_type == 'performance':
                    self.performance_memories[agent_id].clear()
                else:
                    # Clear all memories for this agent
                    self.agent_memories[agent_id].clear()
                    self.performance_memories[agent_id].clear()
                    
                    # Clear embeddings
                    keys_to_remove = [
                        key for key in self.memory_embeddings.keys()
                        if key.startswith(agent_id)
                    ]
                    for key in keys_to_remove:
                        del self.memory_embeddings[key]
                
                logger.info(f"Cleared memories for agent {agent_id}")
            else:
                # Clear all memories
                self.agent_memories.clear()
                self.market_memories.clear()
                self.performance_memories.clear()
                self.memory_embeddings.clear()
                logger.info("Cleared all memories")
                
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
