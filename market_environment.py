# core/market_environment.py
"""
Multi-Agent Market Environment with Dynamic Pricing Simulation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

from models.data_models import (
    MarketConditions, SimulationWeekResult, ProductData, 
    PerformanceMetrics, AgentCommunication
)
from config.settings import config
from agents.llm_agent import LLMIntelligentAgent
from core.rag_system import AdvancedRAGSystem
from core.memory_system import AdvancedMemorySystem

logger = logging.getLogger(__name__)

class MarketSimulator:
    """Simulates market response to pricing decisions"""
    
    def __init__(self):
        self.base_demand_params = {
            'min_demand': 1,
            'base_demand': 20,
            'demand_variance': 10
        }
        
    def simulate_demand(self, decision_price: float, product: ProductData, 
                       market_conditions: MarketConditions) -> float:
        """Simulate market demand response to pricing decision"""
        try:
            # Base demand with exponential distribution
            base_demand = self.base_demand_params['base_demand'] + np.random.exponential(
                self.base_demand_params['demand_variance']
            )
            
            # Price elasticity effect
            price_ratio = decision_price / product.base_price
            elasticity = product.price_elasticity
            demand = base_demand * (price_ratio ** elasticity)
            
            # Market condition effects
            demand *= market_conditions.seasonal_factor
            demand *= market_conditions.economic_indicator
            
            # Competition effect (more competition reduces demand)
            competition_factor = 1.0 - (market_conditions.competition_level * 0.3)
            demand *= competition_factor
            
            # Add volatility
            volatility_factor = 1.0 + np.random.normal(0, market_conditions.demand_volatility)
            demand *= max(0.1, volatility_factor)  # Ensure positive
            
            # Quality and brand effects
            quality_factor = 1.0 + (product.quality_rating - 3.0) * 0.1  # 3 is average
            brand_factor = 1.0 + (product.brand_strength - 5.5) * 0.05  # 5.5 is average
            demand *= quality_factor * brand_factor
            
            return max(self.base_demand_params['min_demand'], demand)
            
        except Exception as e:
            logger.error(f"Error simulating demand: {e}")
            return self.base_demand_params['min_demand']
    
    def calculate_revenue(self, demand: float, price: float) -> float:
        """Calculate revenue from demand and price"""
        return demand * price
    
    def calculate_profit(self, revenue: float, demand: float, cost: float) -> float:
        """Calculate profit from revenue, demand, and unit cost"""
        total_cost = demand * cost
        return revenue - total_cost

class LLMMultiAgentEnvironment:
    """Advanced multi-agent environment with LLM agents and market simulation"""
    
    def __init__(self, products_df: pd.DataFrame, 
                 rag_system: Optional[AdvancedRAGSystem] = None,
                 memory_system: Optional[AdvancedMemorySystem] = None):
        
        self.products_df = products_df.copy()
        self.rag_system = rag_system
        self.memory_system = memory_system
        
        # Environment state
        self.current_week = 0
        self.max_weeks = config.simulation.max_weeks
        self.agents: Dict[str, LLMIntelligentAgent] = {}
        
        # Market simulation
        self.market_simulator = MarketSimulator()
        self.market_conditions = self._initialize_market_conditions()
        
        # Communication system
        self.communication_network: List[AgentCommunication] = []
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.market_history: List[SimulationWeekResult] = []
        self.weekly_revenues = defaultdict(list)
        
        logger.info("LLM Multi-Agent Environment initialized")
    
    def _initialize_market_conditions(self) -> MarketConditions:
        """Initialize market conditions"""
        return MarketConditions(
            competition_level=config.market.base_competition_level,
            demand_volatility=config.market.base_demand_volatility,
            seasonal_factor=1.0,
            economic_indicator=0.6,
            innovation_rate=0.4,
            week=1
        )
    
    def add_agent(self, agent_id: str, agent_type: str, personality: str = "balanced") -> bool:
        """Add LLM agent to environment"""
        try:
            agent = LLMIntelligentAgent(
                agent_id=agent_id,
                agent_type=agent_type,
                personality=personality,
                rag_system=self.rag_system,
                memory_system=self.memory_system
            )
            
            self.agents[agent_id] = agent
            
            # Assign products to agent
            num_products = config.simulation.products_per_agent
            available_products = self.products_df[
                ~self.products_df['product_id'].isin(self._get_assigned_products())
            ]
            
            if len(available_products) < num_products:
                logger.warning(f"Not enough products available for {agent_id}")
                num_products = len(available_products)
            
            if num_products > 0:
                agent_products = available_products.sample(num_products).copy()
                agent.products = agent_products
                
                # Add product knowledge to RAG system
                if self.rag_system:
                    for _, product in agent_products.iterrows():
                        product_text = f"{product['category']} product {product['product_id']} with base price ${product['base_price']:.2f}"
                        self.rag_system.add_product_knowledge(product.to_dict(), product_text)
                
                logger.info(f"Added agent {agent_id} ({agent_type}) with {num_products} products")
                return True
            else:
                logger.error(f"No products available for agent {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding agent {agent_id}: {e}")
            return False
    
    def _get_assigned_products(self) -> List[str]:
        """Get list of already assigned product IDs"""
        assigned = []
        for agent in self.agents.values():
            if agent.products is not None:
                assigned.extend(agent.products['product_id'].tolist())
        return assigned
    
    def _update_market_conditions(self):
        """Update market conditions with seasonal and random factors"""
        try:
            # Seasonal effects (weeks 47-52 are holiday season)
            if self.current_week in range(47, 53):
                seasonal_factor = np.random.uniform(1.1, 1.3)
            elif self.current_week in range(1, 8):  # New year
                seasonal_factor = np.random.uniform(0.8, 0.9)
            else:
                seasonal_factor = np.random.uniform(0.9, 1.1)
            
            # Random market fluctuations with bounds
            competition_change = np.random.normal(0, 0.05)
            new_competition = np.clip(
                self.market_conditions.competition_level + competition_change,
                0.1, 0.9
            )
            
            volatility_change = np.random.normal(0, 0.02)
            new_volatility = np.clip(
                self.market_conditions.demand_volatility + volatility_change,
                0.1, 0.8
            )
            
            economic_change = np.random.normal(0, 0.03)
            new_economic = np.clip(
                self.market_conditions.economic_indicator + economic_change,
                0.2, 0.9
            )
            
            # Update market conditions
            self.market_conditions = MarketConditions(
                competition_level=new_competition,
                demand_volatility=new_volatility,
                seasonal_factor=seasonal_factor,
                economic_indicator=new_economic,
                innovation_rate=np.clip(
                    self.market_conditions.innovation_rate + np.random.normal(0, 0.02),
                    0.2, 0.8
                ),
                week=self.current_week
            )
            
        except Exception as e:
            logger.error(f"Error updating market conditions: {e}")
    
    async def simulate_week(self) -> SimulationWeekResult:
        """Simulate one week of market activity"""
        try:
            self.current_week += 1
            logger.info(f"Simulating week {self.current_week}")
            
            # Update market conditions
            self._update_market_conditions()
            
            # Phase 1: Market Analysis by agents
            market_analyses = {}
            market_context = self.market_conditions.model_dump()
            
            for agent_id, agent in self.agents.items():
                try:
                    analysis = await agent.analyze_market_conditions(market_context)
                    market_analyses[agent_id] = analysis
                except Exception as e:
                    logger.error(f"Market analysis failed for {agent_id}: {e}")
            
            # Phase 2: Inter-agent Communication
            all_communications = []
            for agent_id, agent in self.agents.items():
                try:
                    other_agents = [a for aid, a in self.agents.items() if aid != agent_id]
                    communications = await agent.communicate_with_agents(other_agents, market_context)
                    all_communications.extend(communications)
                except Exception as e:
                    logger.error(f"Communication failed for {agent_id}: {e}")
            
            # Store communications
            self.communication_network.extend(all_communications)
            
            # Phase 3: Pricing Decisions and Market Response
            week_revenues = {}
            week_profits = {}
            total_decisions = 0
            
            for agent_id, agent in self.agents.items():
                agent_revenue = 0.0
                agent_profit = 0.0
                
                if agent.products is not None:
                    for _, product in agent.products.iterrows():
                        try:
                            # Convert product to structured format
                            product_data = ProductData(**product.to_dict())
                            
                            # Make pricing decision
                            decision = await agent.make_pricing_decision(
                                product_data.model_dump(),
                                market_context
                            )
                            
                            # Simulate market response
                            demand = self.market_simulator.simulate_demand(
                                decision.recommended_price,
                                product_data,
                                self.market_conditions
                            )
                            
                            revenue = self.market_simulator.calculate_revenue(
                                demand, decision.recommended_price
                            )
                            
                            profit = self.market_simulator.calculate_profit(
                                revenue, demand, product_data.cost
                            )
                            
                            agent_revenue += revenue
                            agent_profit += profit
                            total_decisions += 1
                            
                            # Update product current price for next iteration
                            agent.products.loc[
                                agent.products['product_id'] == product_data.product_id,
                                'current_price'
                            ] = decision.recommended_price
                            
                        except Exception as e:
                            logger.error(f"Pricing simulation failed for {agent_id}, product {product.get('product_id', 'unknown')}: {e}")
                
                week_revenues[agent_id] = agent_revenue
                week_profits[agent_id] = agent_profit
                
                # Update agent performance
                total_market_revenue = sum(week_revenues.values())
                market_share = (agent_revenue / total_market_revenue * 100) if total_market_revenue > 0 else 0
                
                agent.update_performance_metrics(agent_revenue, market_share)
                self.weekly_revenues[agent_id].append(agent_revenue)
            
            # Phase 4: Performance Analysis and Learning
            await self._analyze_weekly_performance(week_revenues, week_profits)
            
            # Create week result
            week_result = SimulationWeekResult(
                week=self.current_week,
                market_conditions=self.market_conditions,
                agent_revenues=week_revenues,
                total_revenue=sum(week_revenues.values()),
                decisions_made=total_decisions,
                communications_sent=len(all_communications)
            )
            
            self.market_history.append(week_result)
            
            logger.info(f"Week {self.current_week} completed: ${sum(week_revenues.values()):.2f} total revenue, {total_decisions} decisions")
            
            return week_result
            
        except Exception as e:
            logger.error(f"Error simulating week {self.current_week}: {e}")
            # Return empty result
            return SimulationWeekResult(
                week=self.current_week,
                market_conditions=self.market_conditions,
                agent_revenues={},
                total_revenue=0.0,
                decisions_made=0,
                communications_sent=0
            )
    
    async def _analyze_weekly_performance(self, revenues: Dict[str, float], 
                                        profits: Dict[str, float]):
        """Analyze weekly performance and update memories"""
        try:
            total_revenue = sum(revenues.values())
            
            for agent_id, agent in self.agents.items():
                agent_revenue = revenues.get(agent_id, 0)
                agent_profit = profits.get(agent_id, 0)
                
                # Calculate performance metrics
                market_share = (agent_revenue / total_revenue * 100) if total_revenue > 0 else 0
                
                # Calculate price stability (simple measure)
                price_changes = []
                if agent.products is not None:
                    for _, product in agent.products.iterrows():
                        base_price = product['base_price']
                        current_price = product.get('current_price', base_price)
                        change = abs(current_price - base_price) / base_price
                        price_changes.append(change)
                
                price_stability = 1.0 / (1.0 + np.mean(price_changes)) if price_changes else 1.0
                
                performance = PerformanceMetrics(
                    revenue=agent_revenue,
                    market_share=market_share,
                    price_stability=price_stability,
                    customer_satisfaction=np.random.uniform(0.7, 0.9),  # Simulated
                    competitive_position=self._assess_competitive_position(agent_id, revenues),
                    strategic_effectiveness=self._assess_strategy_effectiveness(agent_id)
                )
                
                # Store performance in memory
                if self.memory_system:
                    await self.memory_system.store_performance_metrics(
                        agent_id, performance, self.current_week
                    )
                
                self.performance_history[agent_id].append(performance.model_dump())
                
        except Exception as e:
            logger.error(f"Error analyzing weekly performance: {e}")
    
    def _assess_competitive_position(self, agent_id: str, revenues: Dict[str, float]) -> str:
        """Assess agent's competitive position"""
        try:
            agent_revenue = revenues.get(agent_id, 0)
            sorted_revenues = sorted(revenues.values(), reverse=True)
            
            if not sorted_revenues:
                return "Unknown"
            
            position = sorted_revenues.index(agent_revenue) + 1
            total_agents = len(revenues)
            
            if position == 1:
                return "Leader"
            elif position <= total_agents // 2:
                return "Strong"
            else:
                return "Challenger"
                
        except Exception as e:
            logger.error(f"Error assessing competitive position: {e}")
            return "Unknown"
    
    def _assess_strategy_effectiveness(self, agent_id: str) -> float:
        """Assess strategy effectiveness"""
        try:
            agent = self.agents.get(agent_id)
            if not agent or not agent.decision_history:
                return 0.5
            
            recent_decisions = agent.decision_history[-5:]  # Last 5 decisions
            avg_confidence = np.mean([d.confidence for d in recent_decisions])
            
            # Simple risk-adjusted effectiveness
            risk_balance = np.mean([
                1.0 if d.risk_assessment.value == "Low" else 
                0.7 if d.risk_assessment.value == "Medium" else 0.4 
                for d in recent_decisions
            ])
            
            return (avg_confidence + risk_balance) / 2
            
        except Exception as e:
            logger.error(f"Error assessing strategy effectiveness: {e}")
            return 0.5
    
    def get_environment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        try:
            total_products = len(self.products_df)
            assigned_products = len(self._get_assigned_products())
            
            agent_stats = {}
            for agent_id, agent in self.agents.items():
                agent_stats[agent_id] = agent.get_agent_summary()
            
            return {
                'current_week': self.current_week,
                'total_agents': len(self.agents),
                'total_products': total_products,
                'assigned_products': assigned_products,
                'unassigned_products': total_products - assigned_products,
                'market_conditions': self.market_conditions.model_dump(),
                'total_communications': len(self.communication_network),
                'simulation_weeks': len(self.market_history),
                'agent_statistics': agent_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting environment statistics: {e}")
            return {}
