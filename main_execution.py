# main.py
"""
Main execution file for the LLM Multi-Agent Pricing System
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import config
from simulation.simulator import MARLSimulationSystem
from simulation.data_generator import generate_quick_dataset, get_default_agent_configs
from utils.visualization import create_quick_visualization, export_results_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Main execution function"""
    
    print("🚀 LLM Multi-Agent Pricing System Starting...")
    print("="*60)
    
    try:
        # Validate configuration
        if not config.llm.openai_api_key:
            print("❌ Error: OpenAI API key not configured")
            print("Please set OPENAI_API_KEY environment variable or update config/settings.py")
            return
        
        print("✅ Configuration validated")
        
        # Generate synthetic data
        print("📦 Generating synthetic product data...")
        products_df = generate_quick_dataset(size="medium")  # 50 products
        print(f"   Generated {len(products_df)} products across {products_df['category'].nunique()} categories")
        
        # Get agent configurations
        agent_configs = get_default_agent_configs()
        print(f"   Configured {len(agent_configs)} agents: {[c['agent_type'] for c in agent_configs]}")
        
        # Initialize simulation system
        print("🔧 Initializing simulation system...")
        simulator = MARLSimulationSystem()
        
        # Setup simulation
        success = simulator.setup_simulation(products_df, agent_configs)
        if not success:
            print("❌ Failed to setup simulation")
            return
        
        print("✅ Simulation system ready")
        
        # Run simulation
        print("\n🎮 Starting simulation...")
        print("   This will make LLM API calls for strategic decision making...")
        
        num_weeks = 10  # Shorter simulation for demonstration
        results = await simulator.run_simulation(num_weeks)
        
        if not results:
            print("❌ Simulation failed to produce results")
            return
        
        # Display results
        print("\n📊 Simulation Results:")
        print("="*60)
        
        summary = results.get('simulation_summary', {})
        agent_performance = results.get('agent_performance', {})
        
        print(f"Total Revenue Generated: ${summary.get('total_revenue', 0):,.2f}")
        print(f"Total Decisions Made: {summary.get('total_decisions', 0):,}")
        print(f"Simulation Duration: {summary.get('total_weeks', 0)} weeks")
        
        # Show agent rankings
        if agent_performance:
            print("\n🏆 Agent Performance Rankings:")
            sorted_agents = sorted(
                agent_performance.items(),
                key=lambda x: x[1]['total_revenue'],
                reverse=True
            )
            
            for rank, (agent_id, perf) in enumerate(sorted_agents, 1):
                agent_type = perf.get('agent_type', 'Unknown')
                revenue = perf.get('total_revenue', 0)
                market_share = perf.get('final_market_share', 0)
                decisions = perf.get('decisions_made', 0)
                
                print(f"   #{rank} {agent_id} ({agent_type.upper()})")
                print(f"       Revenue: ${revenue:,.2f} ({market_share:.1f}% market share)")
                print(f"       Decisions: {decisions}")
        
        # Generate analysis report
        print("\n📝 Generating analysis report...")
        report = simulator.generate_analysis_report()
        print(report)
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"simulation_results_{timestamp}.png"
            
            create_quick_visualization(results, viz_path)
            print(f"   Visualizations saved to: {viz_path}")
            
            # Export summary
            summary_path = f"simulation_summary_{timestamp}.txt"
            export_results_summary(results, summary_path)
            print(f"   Summary exported to: {summary_path}")
            
        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
            print("   (Visualization creation failed - continuing without charts)")
        
        # System statistics
        print("\n🔬 System Statistics:")
        print("="*60)
        
        env_stats = results.get('environment_statistics', {})
        rag_stats = results.get('rag_statistics', {})
        memory_stats = results.get('memory_statistics', {})
        
        print(f"Environment:")
        print(f"   Total Agents: {env_stats.get('total_agents', 0)}")
        print(f"   Products Assigned: {env_stats.get('assigned_products', 0)}")
        print(f"   Communications Sent: {env_stats.get('total_communications', 0)}")
        
        print(f"Knowledge System:")
        print(f"   Total Entries: {rag_stats.get('total_entries', 0)}")
        print(f"   Product Knowledge: {rag_stats.get('entries_by_type', {}).get('product', 0)}")
        print(f"   Market Intelligence: {rag_stats.get('entries_by_type', {}).get('market', 0)}")
        
        print(f"Memory System:")
        print(f"   Mem0 Enabled: {memory_stats.get('mem0_enabled', False)}")
        print(f"   Total Memories: {memory_stats.get('total_agent_memories', 0)}")
        print(f"   Performance Records: {memory_stats.get('total_performance_memories', 0)}")
        
        # Success message
        print("\n🎉 Simulation completed successfully!")
        print("   Key Features Demonstrated:")
        print("   ✅ LLM-powered strategic decision making")
        print("   ✅ Multi-agent market simulation")
        print("   ✅ RAG-enhanced knowledge retrieval")
        print("   ✅ Memory-based learning")
        print("   ✅ Agent communication protocols")
        print("   ✅ Performance analysis and visualization")
        
        # Cleanup
        simulator.cleanup()
        
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"❌ Simulation failed: {e}")
        
        if "API" in str(e) or "openai" in str(e).lower():
            print("\n💡 Troubleshooting Tips:")
            print("   • Check your OpenAI API key is valid")
            print("   • Ensure you have sufficient API credits")
            print("   • Verify network connectivity")
            print("   • Try running with fewer agents or shorter duration")

def run_quick_demo():
    """Run a quick demonstration with minimal setup"""
    
    print("🚀 Quick Demo Mode - LLM Multi-Agent Pricing")
    print("="*50)
    
    try:
        # Simple synchronous demo
        from simulation.data_generator import ProductDataGenerator
        from agents.llm_agent import LLMIntelligentAgent
        from models.data_models import AgentType
        
        # Generate sample data
        generator = ProductDataGenerator()
        products_df = generator.generate_products(20)
        print(f"✅ Generated {len(products_df)} sample products")
        
        # Create a single agent for demonstration
        agent = LLMIntelligentAgent("demo_agent", AgentType.ADAPTIVE)
        print("✅ Created adaptive agent")
        
        # Assign some products
        agent.products = products_df.head(5)
        print("✅ Assigned 5 products to agent")
        
        print("\n📊 Sample Product Data:")
        print(products_df.head(3)[['product_id', 'category', 'base_price', 'quality_rating']])
        
        print("\n🎯 Demo completed successfully!")
        print("   To run full simulation, use: python main.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        # Run full simulation
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"Failed to run simulation: {e}")
            print("\nTry running in demo mode: python main.py demo")