# README.md - Usage Guide
"""
LLM Multi-Agent Pricing System - Modular Implementation
======================================================

A sophisticated multi-agent pricing simulation system powered by Large Language Models,
featuring RAG (Retrieval-Augmented Generation), semantic memory, and advanced market dynamics.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run Simulation
```bash
# Full simulation
python main.py

# Quick demo (no API calls)
python main.py demo
```

## Architecture Overview

### Core Modules

1. **config/settings.py** - System configuration and parameters
2. **models/data_models.py** - Pydantic models for structured data
3. **core/rag_system.py** - RAG with FAISS vector search
4. **core/memory_system.py** - Memory management with Mem0 integration
5. **core/market_environment.py** - Market simulation environment
6. **agents/llm_agent.py** - LLM-powered intelligent agents
7. **simulation/simulator.py** - Main simulation controller
8. **simulation/data_generator.py** - Synthetic data generation
9. **utils/visualization.py** - Results visualization

### Key Features

- **LLM-Powered Agents**: Strategic decision making with reasoning
- **RAG System**: Knowledge-augmented pricing decisions
- **Memory Integration**: Learning from past experiences
- **Agent Communication**: Inter-agent coordination protocols
- **Market Simulation**: Realistic demand and competition modeling
- **Performance Analytics**: Comprehensive metrics and visualization

## Usage Examples

### Basic Usage

```python
import asyncio
from simulation.simulator import MARLSimulationSystem
from simulation.data_generator import generate_quick_dataset, get_default_agent_configs

async def run_basic_simulation():
    # Generate data
    products_df = generate_quick_dataset(size="medium")
    agent_configs = get_default_agent_configs()
    
    # Setup simulation
    simulator = MARLSimulationSystem()
    simulator.setup_simulation(products_df, agent_configs)
    
    # Run simulation
    results = await simulator.run_simulation(num_weeks=5)
    
    # Generate report
    report = simulator.generate_analysis_report()
    print(report)

# Run
asyncio.run(run_basic_simulation())
```

### Custom Agent Configuration

```python
from models.data_models import AgentType

# Custom agent setup
custom_agents = [
    {
        'agent_id': 'Aggressive_Trader',
        'agent_type': 'aggressive',
        'personality': 'high_risk'
    },
    {
        'agent_id': 'Market_Maker',
        'agent_type': 'collaborative',
        'personality': 'ecosystem_focused'
    }
]

simulator.setup_simulation(products_df, custom_agents)
```

### Advanced Configuration

```python
from config.settings import config

# Modify configuration
config.llm.model_name = "gpt-4"
config.simulation.max_weeks = 20
config.agents.temperature_settings['aggressive'] = 0.9
config.rag.max_retrievals = 10

# Use custom market scenario
from simulation.data_generator import MarketConditionGenerator

market_gen = MarketConditionGenerator()
volatile_scenario = market_gen.generate_scenario("volatile")
```

### Individual Module Usage

#### RAG System
```python
from core.rag_system import AdvancedRAGSystem

rag = AdvancedRAGSystem()

# Add knowledge
rag.add_product_knowledge(
    {'product_id': 'P001', 'category': 'Electronics', 'price': 299.99},
    "High-end smartphone with premium features"
)

# Retrieve knowledge
results = rag.retrieve_relevant_knowledge("smartphone pricing strategy")
context = rag.get_contextual_prompt("How should I price electronics?")
```

#### Memory System
```python
from core.memory_system import AdvancedMemorySystem
from models.data_models import PricingDecision

memory = AdvancedMemorySystem()

# Store decision
decision = PricingDecision(
    product_id="P001",
    current_price=299.99,
    recommended_price=319.99,
    price_change_pct=6.67,
    reasoning="Market demand increase detected",
    risk_assessment="Medium",
    expected_impact="Positive revenue increase",
    confidence=0.85
)

await memory.store_agent_decision("agent_1", decision, {"market": "strong"})

# Retrieve similar decisions
similar = await memory.retrieve_similar_decisions("agent_1", "electronics pricing")
```

#### LLM Agent
```python
from agents.llm_agent import LLMIntelligentAgent
from models.data_models import AgentType

agent = LLMIntelligentAgent("strategic_agent", AgentType.ADAPTIVE)

# Make market analysis
market_data = {"competition_level": 0.7, "demand_volatility": 0.3}
analysis = await agent.analyze_market_conditions(market_data)

# Make pricing decision
product_data = {"product_id": "P001", "current_price": 299.99, "category": "Electronics"}
decision = await agent.make_pricing_decision(product_data, market_data)
```

### Visualization

```python
from utils.visualization import create_quick_visualization, SimulationVisualizer

# Quick visualization
create_quick_visualization(results, "simulation_charts.png")

# Custom visualization
visualizer = SimulationVisualizer(results)
visualizer.create_comprehensive_dashboard("custom_dashboard.png")
visualizer.create_interactive_dashboard("interactive.html")

# Generate reports
report = visualizer.generate_summary_report()
```

## Configuration Parameters

### LLM Settings
```python
config.llm.model_name = "gpt-4"              # Primary model
config.llm.fallback_model = "gpt-3.5-turbo"  # Fallback model
config.llm.temperature = 0.7                 # Response creativity
config.llm.max_tokens = 1500                 # Response length
config.llm.retry_attempts = 3                # Error retry count
```

### Agent Behavior
```python
config.agents.temperature_settings = {
    "aggressive": 0.8,    # High creativity/risk
    "conservative": 0.3,  # Low creativity/risk
    "adaptive": 0.6,      # Medium creativity
    "collaborative": 0.5  # Balanced approach
}
```

### Market Parameters
```python
config.market.base_competition_level = 0.5
config.market.base_demand_volatility = 0.3
config.market.price_elasticity_range = (-1.5, -0.3)
```

### Simulation Settings
```python
config.simulation.max_weeks = 52
config.simulation.products_per_agent = 5
config.simulation.total_products = 100
```

## Performance Considerations

### API Usage
- Each agent makes 1-2 LLM calls per week per decision
- 4 agents × 5 products × 10 weeks = ~200 API calls
- Monitor API usage and costs accordingly

### Memory Usage
- RAG system stores embeddings in memory
- Memory system accumulates decision history
- Consider cleanup for long-running simulations

### Optimization Tips
- Use smaller product catalogs for testing
- Reduce simulation duration for development
- Enable caching for repeated experiments
- Use fallback models for cost optimization

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: OpenAI API key is required
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Insufficient API Credits**
   ```
   Error: OpenAI API error
   Solution: Check API usage limits and billing
   ```

3. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce products_per_agent or simulation duration
   ```

4. **Import Errors**
   ```
   Error: Module not found
   Solution: Install requirements.txt and check Python path
   ```

### Performance Monitoring

```python
# Check system health
simulator = MARLSimulationSystem()
results = await simulator.run_simulation(5)
health = results['system_health']

print(f"Overall Score: {health['overall_score']}")
print(f"Agents Active: {health['agents_active']}")
print(f"RAG Functional: {health['rag_functional']}")
```

## Research Integration

This implementation extends the original research paper with:

1. **Enhanced Intelligence**: LLM-powered strategic reasoning
2. **Knowledge Management**: RAG for informed decision making
3. **Memory Integration**: Learning from experience
4. **Communication Protocols**: Agent coordination
5. **Advanced Analytics**: Comprehensive performance metrics

## Citation

Based on: "Advanced LLM-Based Multi-Agent Reinforcement Learning System for Dynamic Pricing"

```bibtex
@article{llm_marl_pricing_2024,
  title={Advanced LLM-Based Multi-Agent Reinforcement Learning System for Dynamic Pricing},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

[Specify your license here]

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Support

For issues and questions:
- Check troubleshooting section
- Review configuration parameters
- Monitor API usage and costs
- Test with smaller scenarios first
"""

# Example usage script
if __name__ == "__main__":
    print("LLM Multi-Agent Pricing System - Modular Implementation")
    print("=" * 60)
    print("Architecture:")
    print("  📁 config/          - Configuration management")
    print("  📁 models/          - Data models and schemas")
    print("  📁 core/            - RAG, memory, environment")
    print("  📁 agents/          - LLM-powered agents")
    print("  📁 simulation/      - Simulation control and data")
    print("  📁 utils/           - Visualization and helpers")
    print("  📄 main.py          - Main execution")
    print("  📄 requirements.txt - Dependencies")
    print()
    print("Usage:")
    print("  python main.py      - Run full simulation")
    print("  python main.py demo - Quick demo mode")
    print()
    print("Key Features:")
    print("  ✅ Modular architecture with clean interfaces")
    print("  ✅ LLM-powered strategic decision making")
    print("  ✅ RAG-enhanced knowledge retrieval")
    print("  ✅ Memory-based learning systems")
    print("  ✅ Agent communication protocols")
    print("  ✅ Comprehensive analytics and visualization")
    print("  ✅ Configurable parameters and behaviors")
    print("  ✅ Error handling and fallback mechanisms")
