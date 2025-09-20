# utils/visualization.py
"""
Visualization utilities for simulation results analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimulationVisualizer:
    """Create comprehensive visualizations of simulation results"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.weekly_results = results.get('weekly_results', [])
        self.agent_performance = results.get('agent_performance', {})
        self.simulation_summary = results.get('simulation_summary', {})
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_comprehensive_dashboard(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive dashboard with all key visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LLM Multi-Agent Pricing Simulation Results', fontsize=16, fontweight='bold')
        
        try:
            self._plot_weekly_revenue_trends(axes[0, 0])
            self._plot_market_share_evolution(axes[0, 1])
            self._plot_total_market_revenue(axes[0, 2])
            self._plot_agent_performance_comparison(axes[1, 0])
            self._plot_decision_frequency(axes[1, 1])
            self._plot_communication_activity(axes[1, 2])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dashboard saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            self._create_fallback_visualization(axes)
    
    def _plot_weekly_revenue_trends(self, ax):
        """Plot weekly revenue trends by agent"""
        if not self.weekly_results:
            ax.text(0.5, 0.5, 'No weekly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weekly Revenue Trends')
            return
        
        weeks = [week['week'] for week in self.weekly_results]
        
        for agent_id in self.agent_performance.keys():
            market_shares = []
            for week in self.weekly_results:
                total_revenue = week['total_revenue']
                agent_revenue = week['agent_revenues'].get(agent_id, 0)
                share = (agent_revenue / total_revenue * 100) if total_revenue > 0 else 0
                market_shares.append(share)
            
            agent_type = self.agent_performance[agent_id].get('agent_type', agent_id)
            ax.plot(weeks, market_shares, marker='s', 
                   label=f'{agent_id.replace("_Agent", "")} ({agent_type})', linewidth=2)
        
        ax.set_title('Market Share Evolution', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Market Share (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_total_market_revenue(self, ax):
        """Plot total market revenue per week"""
        if not self.weekly_results:
            ax.text(0.5, 0.5, 'No weekly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Total Market Revenue')
            return
        
        weeks = [week['week'] for week in self.weekly_results]
        total_revenues = [week['total_revenue'] for week in self.weekly_results]
        
        ax.bar(weeks, total_revenues, color='lightblue', alpha=0.7, edgecolor='navy')
        ax.set_title('Total Market Revenue per Week', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Total Revenue ($)')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(weeks) > 1:
            z = np.polyfit(weeks, total_revenues, 1)
            p = np.poly1d(z)
            ax.plot(weeks, p(weeks), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
    
    def _plot_agent_performance_comparison(self, ax):
        """Plot agent performance comparison"""
        if not self.agent_performance:
            ax.text(0.5, 0.5, 'No agent data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Agent Performance')
            return
        
        agents = list(self.agent_performance.keys())
        revenues = [self.agent_performance[agent]['total_revenue'] for agent in agents]
        market_shares = [self.agent_performance[agent]['final_market_share'] for agent in agents]
        
        x_pos = np.arange(len(agents))
        width = 0.35
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos - width/2, revenues, width, label='Total Revenue ($)', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x_pos + width/2, market_shares, width, label='Market Share (%)', alpha=0.8, color='orange')
        
        ax.set_title('Final Agent Performance', fontweight='bold')
        ax.set_xlabel('Agents')
        ax.set_ylabel('Total Revenue ($)', color='blue')
        ax2.set_ylabel('Market Share (%)', color='orange')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([agent.replace('_Agent', '') for agent in agents], rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, revenues):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'${value:,.0f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(bars2, market_shares):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    def _plot_decision_frequency(self, ax):
        """Plot decision frequency over time"""
        if not self.weekly_results:
            ax.text(0.5, 0.5, 'No weekly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Decision Frequency')
            return
        
        weeks = [week['week'] for week in self.weekly_results]
        decisions = [week['decisions_made'] for week in self.weekly_results]
        
        ax.plot(weeks, decisions, marker='o', color='green', linewidth=2)
        ax.fill_between(weeks, decisions, alpha=0.3, color='green')
        ax.set_title('Pricing Decisions per Week', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Decisions Made')
        ax.grid(True, alpha=0.3)
        
        # Add average line
        if decisions:
            avg_decisions = np.mean(decisions)
            ax.axhline(y=avg_decisions, color='red', linestyle='--', 
                      label=f'Average: {avg_decisions:.1f}')
            ax.legend()
    
    def _plot_communication_activity(self, ax):
        """Plot communication activity over time"""
        if not self.weekly_results:
            ax.text(0.5, 0.5, 'No weekly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Communication Activity')
            return
        
        weeks = [week['week'] for week in self.weekly_results]
        communications = [week['communications_sent'] for week in self.weekly_results]
        
        ax.bar(weeks, communications, color='purple', alpha=0.7)
        ax.set_title('Agent Communications per Week', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Messages Sent')
        ax.grid(True, alpha=0.3)
        
        # Add total communications text
        total_comms = sum(communications)
        ax.text(0.02, 0.98, f'Total: {total_comms}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _create_fallback_visualization(self, axes):
        """Create fallback visualization when data is insufficient"""
        for i, ax in enumerate(axes.flat):
            ax.text(0.5, 0.5, f'Insufficient data\nfor Chart {i+1}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Chart {i+1}', fontweight='bold')
    
    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """Create interactive Plotly dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Weekly Revenue Trends', 'Market Share Evolution',
                              'Agent Performance Summary', 'Market Activity'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            if self.weekly_results:
                weeks = [week['week'] for week in self.weekly_results]
                
                # Plot 1: Weekly Revenue Trends
                for agent_id in self.agent_performance.keys():
                    agent_revenues = [week['agent_revenues'].get(agent_id, 0) for week in self.weekly_results]
                    agent_type = self.agent_performance[agent_id].get('agent_type', agent_id)
                    
                    fig.add_trace(
                        go.Scatter(x=weeks, y=agent_revenues, mode='lines+markers',
                                 name=f'{agent_id.replace("_Agent", "")} ({agent_type})'),
                        row=1, col=1
                    )
                
                # Plot 2: Market Share Evolution
                for agent_id in self.agent_performance.keys():
                    market_shares = []
                    for week in self.weekly_results:
                        total_revenue = week['total_revenue']
                        agent_revenue = week['agent_revenues'].get(agent_id, 0)
                        share = (agent_revenue / total_revenue * 100) if total_revenue > 0 else 0
                        market_shares.append(share)
                    
                    fig.add_trace(
                        go.Scatter(x=weeks, y=market_shares, mode='lines+markers',
                                 name=f'{agent_id} Share', showlegend=False),
                        row=1, col=2
                    )
            
            # Plot 3: Agent Performance Summary
            if self.agent_performance:
                agents = list(self.agent_performance.keys())
                revenues = [self.agent_performance[agent]['total_revenue'] for agent in agents]
                market_shares = [self.agent_performance[agent]['final_market_share'] for agent in agents]
                
                fig.add_trace(
                    go.Bar(x=[agent.replace('_Agent', '') for agent in agents], y=revenues,
                          name='Revenue ($)', yaxis='y3'),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=[agent.replace('_Agent', '') for agent in agents], y=market_shares,
                          name='Market Share (%)', yaxis='y4'),
                    row=2, col=1, secondary_y=True
                )
            
            # Plot 4: Market Activity
            if self.weekly_results:
                decisions = [week['decisions_made'] for week in self.weekly_results]
                communications = [week['communications_sent'] for week in self.weekly_results]
                
                fig.add_trace(
                    go.Scatter(x=weeks, y=decisions, mode='lines+markers',
                             name='Decisions', line=dict(color='green')),
                    row=2, col=2
                )
                
                fig.add_trace(
                    go.Bar(x=weeks, y=communications, name='Communications',
                          marker_color='purple', opacity=0.7),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="LLM Multi-Agent Pricing Simulation - Interactive Dashboard",
                showlegend=True,
                height=800
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to {save_path}")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
    
    def generate_summary_report(self) -> str:
        """Generate text summary report"""
        try:
            if not self.simulation_summary:
                return "No simulation summary available."
            
            summary = self.simulation_summary
            
            report = f"""
SIMULATION PERFORMANCE SUMMARY
{'='*50}

Duration: {summary.get('total_weeks', 0)} weeks
Total Market Revenue: ${summary.get('total_revenue', 0):,.2f}
Average Weekly Revenue: ${summary.get('avg_weekly_revenue', 0):,.2f}
Total Decisions: {summary.get('total_decisions', 0):,}
Communications: {summary.get('total_communications', 0):,}

AGENT RANKINGS:
"""
            
            # Sort agents by performance
            if self.agent_performance:
                sorted_agents = sorted(
                    self.agent_performance.items(),
                    key=lambda x: x[1]['total_revenue'],
                    reverse=True
                )
                
                for rank, (agent_id, perf) in enumerate(sorted_agents, 1):
                    agent_type = perf.get('agent_type', 'Unknown')
                    revenue = perf.get('total_revenue', 0)
                    market_share = perf.get('final_market_share', 0)
                    
                    report += f"""
#{rank} {agent_id} ({agent_type.upper()})
    Revenue: ${revenue:,.2f} ({market_share:.1f}% market share)
    Decisions: {perf.get('decisions_made', 0)}
    Communications: {perf.get('communication_activity', 0)}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return f"Error generating report: {e}"

def create_quick_visualization(results: Dict[str, Any], save_path: str = None):
    """Quick function to create standard visualization"""
    visualizer = SimulationVisualizer(results)
    visualizer.create_comprehensive_dashboard(save_path)
    return visualizer

def export_results_summary(results: Dict[str, Any], filepath: str):
    """Export results summary to text file"""
    try:
        visualizer = SimulationVisualizer(results)
        report = visualizer.generate_summary_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Results summary exported to {filepath}")
        
    except Exception as e:
        logger.error(f"Error exporting results summary: {e}")agent_revenues = [week['agent_revenues'].get(agent_id, 0) for week in self.weekly_results]
            agent_type = self.agent_performance[agent_id].get('agent_type', agent_id)
            
            ax.plot(weeks, agent_revenues, marker='o', 
                   label=f'{agent_id.replace("_Agent", "")} ({agent_type})', linewidth=2)
        
        ax.set_title('Weekly Revenue by Agent', fontweight='bold')
        ax.set_xlabel('Week')
        ax.set_ylabel('Revenue ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_market_share_evolution(self, ax):
        """Plot market share evolution over time"""
        if not self.weekly_results:
            ax.text(0.5, 0.5, 'No weekly data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Market Share Evolution')
            return
        
        weeks = [week['week'] for week in self.weekly_results]
        
        for agent_id in self.agent_performance.keys():
            