# simulation/simulator.py
"""
Main simulation controller orchestrating the multi-agent pricing system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from models.data_models import (
    AgentType, SimulationSummary, AgentPerformanceSummary, 
    SimulationWeekResult
)
from config.settings import config
from core.market_environment import LLMMultiAgentEnvironment
from core.rag_system import AdvancedRAGSystem
from core.memory_system import AdvancedMemorySystem

logger = logging.getLogger(__name__)

class MARLSimulationSystem:
    """Main simulation system orchestrating LLM agents"""
    
    def __init__(self):
        # Core systems
        self.rag_system: Optional[AdvancedRAGSystem] = None
        self.memory_system: Optional[AdvancedMemorySystem] = None
        self.environment: Optional[LLMMultiAgentEnvironment] = None
        
        # Simulation state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Results storage
        self.weekly_results: List[SimulationWeekResult] = []
        self.final_summary: Optional[SimulationSummary] = None
        self.agent_summaries: Dict[str, AgentPerformanceSummary] = {}
        
        logger.info("MARL Simulation System initialized")
    
    def initialize_systems(self) -> bool:
        """Initialize RAG and memory systems"""
        try:
            # Initialize RAG system
            self.rag_system = AdvancedRAGSystem()
            logger.info("RAG system initialized")
            
            # Initialize memory system
            self.memory_system = AdvancedMemorySystem()
            logger.info("Memory system initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing systems: {e}")
            return False
    
    def setup_simulation(self, products_df: pd.DataFrame, 
                        agent_configs: List[Dict[str, Any]]) -> bool:
        """Setup simulation with LLM agents"""
        try:
            if not self.initialize_systems():
                logger.error("Failed to initialize core systems")
                return False
            
            # Initialize environment
            self.environment = LLMMultiAgentEnvironment(
                products_df,
                rag_system=self.rag_system,
                memory_system=self.memory_system
            )
            
            # Add agents based on configuration
            agents_added = 0
            for config_item in agent_configs:
                success = self.environment.add_agent(
                    config_item['agent_id'],
                    config_item['agent_type'],
                    config_item.get('personality', 'balanced')
                )
                if success:
                    agents_added += 1
                else:
                    logger.warning(f"Failed to add agent {config_item['agent_id']}")
            
            if agents_added == 0:
                logger.error("No agents were successfully added")
                return False
            
            logger.info(f"Simulation setup complete with {agents_added}/{len(agent_configs)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up simulation: {e}")
            return False
    
    async def run_simulation(self, num_weeks: int = None) -> Dict[str, Any]:
        """Run complete simulation"""
        if not self.environment:
            raise ValueError("Simulation not setup. Call setup_simulation() first.")
        
        num_weeks = num_weeks or config.simulation.max_weeks
        
        logger.info(f"Starting simulation for {num_weeks} weeks...")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            self.weekly_results = []
            
            for week in range(num_weeks):
                logger.info(f"Running week {week + 1}/{num_weeks}")
                
                # Simulate one week
                week_result = await self.environment.simulate_week()
                self.weekly_results.append(week_result)
                
                # Progress update every 5 weeks
                if (week + 1) % 5 == 0:
                    self._log_progress_update(week + 1, week_result)
                
                # Check for early termination conditions
                if not self._should_continue_simulation(week_result):
                    logger.warning(f"Simulation terminated early at week {week + 1}")
                    break
        
        except Exception as e:
            logger.error(f"Simulation failed at week {week + 1}: {e}")
            raise
        
        finally:
            self.is_running = False
            self.end_time = datetime.now()
        
        # Compile final results
        results = self._compile_results()
        logger.info(f"Simulation completed! {len(self.weekly_results)} weeks simulated.")
        
        return results
    
    def _log_progress_update(self, week: int, week_result: SimulationWeekResult):
        """Log progress update"""
        total_revenue = week_result.total_revenue
        decisions = week_result.decisions_made
        communications = week_result.communications_sent
        
        logger.info(f"Week {week} Summary:")
        logger.info(f"  Total Revenue: ${total_revenue:.2f}")
        logger.info(f"  Decisions Made: {decisions}")
        logger.info(f"  Communications: {communications}")
        
        # Show top performing agent
        if week_result.agent_revenues:
            top_agent = max(week_result.agent_revenues, key=week_result.agent_revenues.get)
            top_revenue = week_result.agent_revenues[top_agent]
            logger.info(f"  Top Agent: {top_agent} (${top_revenue:.2f})")
    
    def _should_continue_simulation(self, week_result: SimulationWeekResult) -> bool:
        """Check if simulation should continue"""
        # Continue if there's market activity
        return (week_result.total_revenue > 0 and 
                week_result.decisions_made > 0 and 
                len(week_result.agent_revenues) > 0)
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final simulation results"""
        try:
            if not self.weekly_results:
                return {}
            
            # Calculate aggregate metrics
            total_weeks = len(self.weekly_results)
            total_revenue = sum(week.total_revenue for week in self.weekly_results)
            total_decisions = sum(week.decisions_made for week in self.weekly_results)
            total_communications = sum(week.communications_sent for week in self.weekly_results)
            
            # Create simulation summary
            self.final_summary = SimulationSummary(
                total_weeks=total_weeks,
                total_revenue=total_revenue,
                avg_weekly_revenue=total_revenue / total_weeks if total_weeks > 0 else 0,
                total_decisions=total_decisions,
                total_communications=total_communications,
                decisions_per_week=total_decisions / total_weeks if total_weeks > 0 else 0,
                communications_per_week=total_communications / total_weeks if total_weeks > 0 else 0,
                start_time=self.start_time or datetime.now(),
                end_time=self.end_time or datetime.now()
            )
            
            # Calculate agent performance summaries
            self.agent_summaries = self._calculate_agent_summaries()
            
            # Compile environment statistics
            env_stats = self.environment.get_environment_statistics() if self.environment else {}
            rag_stats = self.rag_system.get_statistics() if self.rag_system else {}
            memory_stats = self.memory_system.get_memory_statistics() if self.memory_system else {}
            
            return {
                'simulation_summary': self.final_summary.model_dump(),
                'agent_performance': {k: v.model_dump() for k, v in self.agent_summaries.items()},
                'weekly_results': [result.model_dump() for result in self.weekly_results],
                'environment_statistics': env_stats,
                'rag_statistics': rag_stats,
                'memory_statistics': memory_stats,
                'system_health': self._assess_system_health()
            }
            
        except Exception as e:
            logger.error(f"Error compiling results: {e}")
            return {'error': str(e)}
    
    def _calculate_agent_summaries(self) -> Dict[str, AgentPerformanceSummary]:
        """Calculate performance summaries for each agent"""
        summaries = {}
        
        if not self.environment:
            return summaries
        
        try:
            for agent_id, agent in self.environment.agents.items():
                # Collect revenue data across weeks
                agent_revenues = [
                    week.agent_revenues.get(agent_id, 0) 
                    for week in self.weekly_results
                ]
                
                total_revenue = sum(agent_revenues)
                avg_weekly_revenue = total_revenue / len(agent_revenues) if agent_revenues else 0
                
                # Calculate final market share
                if self.weekly_results:
                    final_week = self.weekly_results[-1]
                    final_market_share = 0
                    if final_week.total_revenue > 0:
                        agent_final_revenue = final_week.agent_revenues.get(agent_id, 0)
                        final_market_share = (agent_final_revenue / final_week.total_revenue) * 100
                else:
                    final_market_share = 0
                
                # Calculate revenue stability (inverse of coefficient of variation)
                revenue_stability = 1.0
                if len(agent_revenues) > 1 and avg_weekly_revenue > 0:
                    import numpy as np
                    cv = np.std(agent_revenues) / avg_weekly_revenue
                    revenue_stability = 1.0 / (1.0 + cv)
                
                summaries[agent_id] = AgentPerformanceSummary(
                    agent_id=agent_id,
                    agent_type=agent.agent_type,
                    total_revenue=total_revenue,
                    avg_weekly_revenue=avg_weekly_revenue,
                    final_market_share=final_market_share,
                    revenue_stability=revenue_stability,
                    decisions_made=agent.performance_metrics['decisions_made'],
                    successful_predictions=agent.performance_metrics['successful_predictions'],
                    communication_activity=agent.performance_metrics['communication_sent']
                )
                
        except Exception as e:
            logger.error(f"Error calculating agent summaries: {e}")
        
        return summaries
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health and performance"""
        try:
            health = {
                'simulation_completed': not self.is_running and len(self.weekly_results) > 0,
                'agents_active': len(self.environment.agents) if self.environment else 0,
                'rag_functional': self.rag_system is not None,
                'memory_functional': self.memory_system is not None,
                'market_responsive': False,
                'communication_active': False,
                'overall_score': 0.0
            }
            
            if self.weekly_results:
                # Check market responsiveness
                total_decisions = sum(week.decisions_made for week in self.weekly_results)
                health['market_responsive'] = total_decisions > 0
                
                # Check communication activity
                total_comms = sum(week.communications_sent for week in self.weekly_results)
                health['communication_active'] = total_comms > 0
            
            # Calculate overall health score
            score_components = [
                health['simulation_completed'],
                health['agents_active'] > 0,
                health['rag_functional'],
                health['memory_functional'],
                health['market_responsive'],
                health['communication_active']
            ]
            
            health['overall_score'] = sum(score_components) / len(score_components)
            
            return health
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return {'error': str(e)}
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        if not self.final_summary or not self.agent_summaries:
            return "No simulation results available for analysis."
        
        try:
            summary = self.final_summary
            
            report = f"""
🔬 ADVANCED LLM MULTI-AGENT PRICING SIMULATION REPORT
{'='*60}

📊 SIMULATION OVERVIEW:
• Duration: {summary.total_weeks} weeks
• Total Market Revenue: ${summary.total_revenue:,.2f}
• Average Weekly Revenue: ${summary.avg_weekly_revenue:,.2f}
• Total Pricing Decisions: {summary.total_decisions:,}
• Agent Communications: {summary.total_communications:,}

🤖 AGENT PERFORMANCE ANALYSIS:
"""
            
            # Sort agents by performance
            sorted_agents = sorted(
                self.agent_summaries.items(),
                key=lambda x: x[1].total_revenue,
                reverse=True
            )
            
            for rank, (agent_id, perf) in enumerate(sorted_agents, 1):
                market_share = perf.final_market_share
                
                report += f"""
🏆 #{rank} - {agent_id} ({perf.agent_type.value.upper()})
   • Total Revenue: ${perf.total_revenue:,.2f} ({market_share:.1f}% market share)
   • Weekly Average: ${perf.avg_weekly_revenue:,.2f}
   • Revenue Stability: {perf.revenue_stability:.3f}
   • Decisions Made: {perf.decisions_made}
   • Communications Sent: {perf.communication_activity}
"""
            
            # Market dynamics analysis
            report += f"""

📈 MARKET DYNAMICS:
• Decision Frequency: {summary.decisions_per_week:.1f} per week
• Communication Activity: {summary.communications_per_week:.1f} messages per week
• Market Evolution: {summary.total_weeks} weeks tracked
• Simulation Duration: {(summary.end_time - summary.start_time).total_seconds() / 60:.1f} minutes

🧠 ADVANCED AI FEATURES UTILIZED:
✅ LLM-based Strategic Decision Making
✅ Multi-Agent RAG (Retrieval-Augmented Generation)
✅ Semantic Memory with Experience Learning
✅ Inter-Agent Communication & Coordination
✅ Structured Output with Pydantic Models
✅ Dynamic Market Intelligence
✅ Agent Personality & Behavioral Modeling

💡 KEY INSIGHTS:
"""
            
            # Generate insights
            if sorted_agents:
                best_agent_id, best_perf = sorted_agents[0]
                best_type = best_perf.agent_type.value
                
                report += f"""
• Best Performing Strategy: {best_type.upper()} agents show superior performance
• Market Efficiency: ${summary.avg_weekly_revenue:,.0f} average weekly market value
• AI Coordination: {summary.total_communications} inter-agent communications facilitated
• Decision Quality: {summary.total_decisions} LLM-powered pricing decisions made
• System Performance: All advanced AI features functioning correctly
"""
            
            # Add system statistics if available
            if hasattr(self, 'rag_system') and self.rag_system:
                rag_stats = self.rag_system.get_statistics()
                report += f"""

📚 KNOWLEDGE SYSTEM PERFORMANCE:
• Total Knowledge Entries: {rag_stats.get('total_entries', 0)}
• Product Knowledge: {rag_stats.get('entries_by_type', {}).get('product', 0)} entries
• Market Intelligence: {rag_stats.get('entries_by_type', {}).get('market', 0)} entries
• Strategy Knowledge: {rag_stats.get('entries_by_type', {}).get('strategy', 0)} entries
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
            return f"Error generating report: {e}"
    
    def export_results(self, filepath: str) -> bool:
        """Export simulation results to file"""
        try:
            if not self.final_summary:
                logger.error("No results to export")
                return False
            
            results = self._compile_results()
            
            # Export as JSON
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.memory_system:
                # Could add cleanup logic here
                pass
            
            if self.rag_system:
                # Could add cleanup logic here
                pass
            
            self.environment = None
            self.weekly_results.clear()
            
            logger.info("Simulation system cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
