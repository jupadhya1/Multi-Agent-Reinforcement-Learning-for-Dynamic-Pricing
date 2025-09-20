# agents/llm_agent.py
"""
LLM-powered intelligent pricing agents with reasoning capabilities
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import openai
from openai import OpenAI

from models.data_models import (
    AgentType, MarketAnalysis, PricingDecision, AgentCommunication,
    MessageType, Priority, RiskLevel
)
from config.settings import config
from core.rag_system import AdvancedRAGSystem
from core.memory_system import AdvancedMemorySystem

logger = logging.getLogger(__name__)

class LLMIntelligentAgent:
    """Advanced LLM-based pricing agent with RAG and memory integration"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, personality: str = "balanced",
                 rag_system: Optional[AdvancedRAGSystem] = None,
                 memory_system: Optional[AdvancedMemorySystem] = None):
        
        self.agent_id = agent_id
        self.agent_type = AgentType(agent_type)
        self.personality = personality
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.llm.openai_api_key)
        
        # External systems
        self.rag_system = rag_system
        self.memory_system = memory_system
        
        # Agent configuration
        self.temperature = self._get_temperature_by_type()
        self.model_name = config.llm.model_name
        self.fallback_model = config.llm.fallback_model
        
        # Agent state
        self.decision_history = []
        self.communication_log = []
        self.products = None  # Will be assigned by environment
        
        # Performance tracking
        self.performance_metrics = {
            'total_revenue': 0.0,
            'decisions_made': 0,
            'successful_predictions': 0,
            'market_share': 0.0,
            'communication_sent': 0
        }
        
        logger.info(f"LLM Agent {agent_id} ({agent_type.value}) initialized")
    
    def _get_temperature_by_type(self) -> float:
        """Set LLM temperature based on agent type"""
        return config.agents.temperature_settings.get(self.agent_type.value, 0.5)
    
    def _get_personality_traits(self) -> Dict[str, float]:
        """Get personality traits for this agent type"""
        return config.agents.personality_traits.get(self.agent_type.value, {})
    
    async def _make_llm_request(self, messages: List[Dict[str, str]], 
                               max_tokens: int = None, temperature: float = None) -> str:
        """Make LLM request with error handling and fallbacks"""
        max_tokens = max_tokens or config.llm.max_tokens
        temperature = temperature or self.temperature
        
        for attempt in range(config.llm.retry_attempts):
            try:
                # Try primary model first
                model = self.model_name if attempt == 0 else self.fallback_model
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=config.llm.timeout
                )
                
                return response.choices[0].message.content
                
            except openai.APIError as e:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt == config.llm.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error in LLM request: {e}")
                if attempt == config.llm.retry_attempts - 1:
                    raise
                await asyncio.sleep(1)
        
        raise Exception("All LLM request attempts failed")
    
    def _parse_json_response(self, response_text: str, fallback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON response with fallback handling"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for {self.agent_id}, using fallback")
            return fallback_data
    
    async def analyze_market_conditions(self, market_data: Dict[str, Any]) -> MarketAnalysis:
        """Analyze market conditions using LLM with RAG"""
        try:
            # Retrieve relevant knowledge if RAG system available
            context_prompt = ""
            if self.rag_system:
                query = f"Market analysis for week {market_data.get('week', 'current')} with competition {market_data.get('competition_level', 0.5)}"
                context_prompt = self.rag_system.get_contextual_prompt(query, "market")
            
            # Get similar past analyses from memory
            past_analysis = ""
            if self.memory_system:
                similar_decisions = await self.memory_system.retrieve_similar_decisions(
                    self.agent_id, f"market analysis {market_data.get('week', '')}", k=3
                )
                if similar_decisions:
                    past_analysis = "\n".join([
                        f"Past analysis: {d.get('content', d.get('decision', {}).get('reasoning', 'No reasoning'))}"
                        for d in similar_decisions[:2]
                    ])
            
            # Get personality traits
            traits = self._get_personality_traits()
            
            system_prompt = f"""
You are a {self.agent_type.value} pricing agent with {self.personality} personality.
Your role is to analyze market conditions and provide strategic insights.

Agent Type Characteristics:
- Aggressive: Take bold pricing actions, focus on market capture, high risk tolerance
- Conservative: Prioritize stability and risk mitigation, avoid volatility
- Adaptive: Adjust strategy based on market feedback, learning-oriented
- Collaborative: Consider impact on market ecosystem, cooperation-focused

Your Personality Traits: {traits}

Your Past Experience:
{past_analysis}

Analyze the current market situation and provide structured insights.
Respond with valid JSON only.
"""

            user_prompt = f"""
{context_prompt}

Current Market Data:
{json.dumps(market_data, indent=2)}

Analyze these market conditions and provide a JSON response with:
{{
    "market_trend": "string (rising/falling/stable/volatile)",
    "competition_level": number (0.0-1.0),
    "demand_forecast": "string (increasing/decreasing/stable)",
    "price_sensitivity": number (0.0-1.0),
    "recommended_strategy": "string (strategy recommendation)",
    "confidence_score": number (0.0-1.0),
    "key_insights": ["insight1", "insight2", "insight3"]
}}

Consider your {self.agent_type.value} nature and provide analysis accordingly.
"""

            # Make LLM request
            response_text = await self._make_llm_request([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Parse response with fallback
            fallback_data = {
                'market_trend': 'stable',
                'competition_level': market_data.get('competition_level', 0.5),
                'demand_forecast': 'stable',
                'price_sensitivity': 0.5,
                'recommended_strategy': f'{self.agent_type.value} approach',
                'confidence_score': 0.7,
                'key_insights': [f'Market analysis by {self.agent_type.value} agent']
            }
            
            response_data = self._parse_json_response(response_text, fallback_data)
            
            analysis = MarketAnalysis(
                market_trend=response_data.get('market_trend', 'stable'),
                competition_level=max(0, min(1, response_data.get('competition_level', 0.5))),
                demand_forecast=response_data.get('demand_forecast', 'stable'),
                price_sensitivity=max(0, min(1, response_data.get('price_sensitivity', 0.5))),
                recommended_strategy=response_data.get('recommended_strategy', 'maintain'),
                confidence_score=max(0, min(1, response_data.get('confidence_score', 0.7))),
                key_insights=response_data.get('key_insights', [])[:10]  # Limit insights
            )
            
            # Store analysis in memory and RAG if available
            if self.memory_system:
                await self.memory_system.store_market_intelligence(analysis, self.agent_id)
            
            if self.rag_system:
                self.rag_system.add_market_intelligence(
                    response_data,
                    f"Market analysis: {analysis.market_trend} trend with {analysis.recommended_strategy} strategy"
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed for {self.agent_id}: {e}")
            # Return default analysis
            return MarketAnalysis(
                market_trend="stable",
                competition_level=0.5,
                demand_forecast="stable",
                price_sensitivity=0.5,
                recommended_strategy="maintain current approach",
                confidence_score=0.5,
                key_insights=[f"Analysis failed for {self.agent_id}, using defaults"]
            )
    
    async def make_pricing_decision(self, product_data: Dict[str, Any], 
                                   market_context: Dict[str, Any]) -> PricingDecision:
        """Make pricing decision using LLM with full context"""
        try:
            # Retrieve relevant knowledge
            context_prompt = ""
            if self.rag_system:
                query = f"Pricing decision for {product_data.get('category', 'unknown')} product price {product_data.get('current_price', 0)}"
                context_prompt = self.rag_system.get_contextual_prompt(query, "all")
            
            # Get agent's performance trends
            performance_trends = {}
            if self.memory_system:
                performance_trends = await self.memory_system.get_agent_performance_trends(self.agent_id)
            
            # Get similar past decisions
            similar_decisions = []
            if self.memory_system:
                similar_decisions = await self.memory_system.retrieve_similar_decisions(
                    self.agent_id, query, k=3
                )
            
            past_decisions = ""
            if similar_decisions:
                past_decisions = "\n".join([
                    f"Past: {d.get('content', d.get('decision', {}).get('reasoning', 'No reasoning'))} -> Result: {d.get('context', {}).get('outcome', 'Unknown')}"
                    for d in similar_decisions[:2]
                ])
            
            traits = self._get_personality_traits()
            
            system_prompt = f"""
You are a {self.agent_type.value} pricing agent making strategic pricing decisions.

Your Performance History:
- Revenue trend: {performance_trends.get('revenue_trend', 0):.3f}
- Average market share: {performance_trends.get('avg_market_share', 0):.1f}%
- Total decisions made: {performance_trends.get('total_decisions', 0)}

Your Personality Traits: {traits}

Your Past Similar Decisions:
{past_decisions}

Agent Behavior Guidelines:
- Aggressive: Take bold pricing moves, aim for market leadership, high price changes
- Conservative: Focus on stability, avoid risky price changes, small adjustments
- Adaptive: Adjust based on market feedback and trends, moderate changes
- Collaborative: Consider market ecosystem health, coordination-friendly

Make a pricing decision that aligns with your {self.agent_type.value} type.
Respond with valid JSON only.
"""

            user_prompt = f"""
{context_prompt}

Product Information:
{json.dumps(product_data, indent=2)}

Market Context:
{json.dumps(market_context, indent=2)}

Make a pricing decision with JSON response:
{{
    "recommended_price": number,
    "price_change_pct": number,
    "reasoning": "detailed reasoning string",
    "risk_assessment": "Low/Medium/High",
    "expected_impact": "string describing expected impact",
    "confidence": number (0.0-1.0)
}}

Consider your {self.agent_type.value} nature and the current market conditions.
"""

            # Make LLM request
            response_text = await self._make_llm_request([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # Create fallback decision based on agent type
            current_price = product_data.get('current_price', product_data.get('base_price', 100))
            
            if self.agent_type == AgentType.AGGRESSIVE:
                new_price = current_price * 1.05  # 5% increase
                risk = "High"
            elif self.agent_type == AgentType.CONSERVATIVE:
                new_price = current_price * 1.01  # 1% increase
                risk = "Low"
            elif self.agent_type == AgentType.ADAPTIVE:
                change_factor = 1.02 + (market_context.get('competition_level', 0.5) * 0.03)
                new_price = current_price * change_factor
                risk = "Medium"
            else:  # COLLABORATIVE
                new_price = current_price * 1.015  # 1.5% increase
                risk = "Low"
            
            fallback_data = {
                'recommended_price': new_price,
                'price_change_pct': (new_price - current_price) / current_price * 100,
                'reasoning': f'{self.agent_type.value} pricing strategy applied based on market conditions',
                'risk_assessment': risk,
                'expected_impact': f'Moderate impact expected from {self.agent_type.value} strategy',
                'confidence': 0.7
            }
            
            response_data = self._parse_json_response(response_text, fallback_data)
            
            decision = PricingDecision(
                product_id=product_data.get('product_id', 'unknown'),
                current_price=float(product_data.get('current_price', product_data.get('base_price', 0))),
                recommended_price=float(response_data.get('recommended_price', new_price)),
                price_change_pct=float(response_data.get('price_change_pct', 0)),
                reasoning=str(response_data.get('reasoning', 'No reasoning provided')),
                risk_assessment=RiskLevel(response_data.get('risk_assessment', 'Medium')),
                expected_impact=str(response_data.get('expected_impact', 'Neutral')),
                confidence=max(0, min(1, float(response_data.get('confidence', 0.7))))
            )
            
            # Store decision in memory
            if self.memory_system:
                await self.memory_system.store_agent_decision(self.agent_id, decision, market_context)
            
            # Add to decision history
            self.decision_history.append(decision)
            self.performance_metrics['decisions_made'] += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Pricing decision failed for {self.agent_id}: {e}")
            # Return safe default decision
            current_price = float(product_data.get('current_price', product_data.get('base_price', 100)))
            return PricingDecision(
                product_id=product_data.get('product_id', 'unknown'),
                current_price=current_price,
                recommended_price=current_price,
                price_change_pct=0.0,
                reasoning=f"Decision failed for {self.agent_id}, maintaining current price",
                risk_assessment=RiskLevel.LOW,
                expected_impact="Neutral - no change",
                confidence=0.5
            )
    
    async def communicate_with_agents(self, other_agents: List['LLMIntelligentAgent'], 
                                    market_context: Dict[str, Any]) -> List[AgentCommunication]:
        """Communicate with other agents based on agent type"""
        
        # Only collaborative agents communicate actively by default
        if self.agent_type != AgentType.COLLABORATIVE:
            return []
        
        try:
            system_prompt = f"""
You are a collaborative pricing agent. Your role is to share insights and coordinate 
with other agents to maintain market stability while optimizing collective performance.

Consider the current market context and decide what information to share with other agents.
Focus on:
1. Market insights that benefit the ecosystem
2. Coordination opportunities  
3. Risk warnings
4. Strategic suggestions

Be diplomatic and constructive in your communications.
Respond with valid JSON only.
"""
            
            user_prompt = f"""
Current market context:
{json.dumps(market_context, indent=2)}

Other agents in the market: {[agent.agent_id for agent in other_agents]}

What messages would you like to send to coordinate with other agents?
Provide JSON response:
{{
    "messages": [
        {{
            "receiver": "agent_id or 'broadcast'",
            "type": "strategy_share/market_info/coordination/warning",
            "content": {{"message": "content", "data": "any_data"}},
            "priority": "low/medium/high"
        }}
    ]
}}
"""
            
            # Make LLM request
            response_text = await self._make_llm_request([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=800)
            
            # Parse response with fallback
            fallback_data = {
                'messages': [{
                    'receiver': 'broadcast',
                    'type': 'coordination',
                    'content': {'message': f'Coordination message from {self.agent_id}'},
                    'priority': 'medium'
                }]
            }
            
            response_data = self._parse_json_response(response_text, fallback_data)
            
            # Create communication messages
            communications = []
            if 'messages' in response_data:
                for msg_data in response_data['messages'][:5]:  # Limit to 5 messages
                    try:
                        communication = AgentCommunication(
                            sender_id=self.agent_id,
                            receiver_id=str(msg_data.get('receiver', 'broadcast')),
                            message_type=MessageType(msg_data.get('type', 'coordination')),
                            content=msg_data.get('content', {}),
                            priority=Priority(msg_data.get('priority', 'medium'))
                        )
                        communications.append(communication)
                    except ValueError as e:
                        logger.warning(f"Invalid communication data: {e}")
                        continue
            
            self.communication_log.extend(communications)
            self.performance_metrics['communication_sent'] += len(communications)
            
            return communications
            
        except Exception as e:
            logger.error(f"Communication failed for {self.agent_id}: {e}")
            return []
    
    def update_performance_metrics(self, revenue: float, market_share: float):
        """Update agent performance metrics"""
        self.performance_metrics['total_revenue'] += revenue
        self.performance_metrics['market_share'] = market_share
        
        # Simple success prediction based on positive revenue
        if revenue > 0:
            self.performance_metrics['successful_predictions'] += 1
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent summary"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'personality': self.personality,
            'temperature': self.temperature,
            'decisions_made': len(self.decision_history),
            'communications_sent': len(self.communication_log),
            'performance_metrics': self.performance_metrics.copy(),
            'products_assigned': len(self.products) if self.products is not None else 0,
            'latest_decision': self.decision_history[-1].model_dump() if self.decision_history else None
        }
                '