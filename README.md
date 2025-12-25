# 🛒 Multi-Agent Reinforcement Learning for Dynamic Pricing


## Overview

This project demonstrates an advanced multi-agent system designed for complex retail decision-making, specifically focusing on **dynamic pricing optimization**. It showcases an enterprise-grade architecture with a strong emphasis on **observability**, using LangSmith for end-to-end tracing and monitoring.

The system leverages multiple specialized AI agents that collaborate through a sophisticated orchestration layer, utilizing the **Model Context Protocol (MCP)** for tool access and **Agent-to-Agent (A2A)** communication for seamless information sharing.

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent System** | Team of specialized AI agents collaborating to solve complex retail pricing problems |
| **Model Context Protocol (MCP)** | Simulated MCP server providing domain-specific tools like `price_optimizer` and `demand_forecaster` |
| **A2A Communication** | Dedicated layer for agents to interact, share insights, and broadcast information |
| **Observable Vector Database** | ChromaDB-powered vector store enriching agents with business knowledge and policies |
| **End-to-End Observability** | LangSmith integration for detailed traces of every agent action and LLM call |
| **Workflow Orchestration** | Central orchestrator managing complex workflows and aggregating findings |

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ObservableRetailOrchestrator                 │
│                    (Workflow Management & Aggregation)          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌─────────────────┐  ┌───────────────┐  ┌─────────────────┐
│  Pricing Agent  │  │ Pricing Agent │  │  Vector DB      │
│       #1        │  │      #2       │  │ (Market Context)│
└────────┬────────┘  └───────┬───────┘  └─────────────────┘
         │                   │
         │    ┌──────────────┴──────────────┐
         │    │                             │
         ▼    ▼                             ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│     MCP Tool Server     │      │   A2A Communication     │
│  • price_optimizer      │      │       Layer             │
│  • demand_forecaster    │      │  (Inter-agent messaging)│
└─────────────────────────┘      └─────────────────────────┘
```

### Component Descriptions

| Component | Responsibility |
|-----------|----------------|
| `ObservableRetailOrchestrator` | The "brain" of the operation—initiates workflows, delegates tasks, and aggregates recommendations |
| `EnhancedRetailAgent` | Base class blueprint for all agents with MCP, A2A, and Vector DB connections |
| `EnhancedPricingAgent` | Specialized agent for pricing analysis using tools and knowledge base |
| `RetailMCPServer` | Collection of high-power tools for complex calculations and analyses |
| `A2ACommunicationLayer` | Manages all inter-agent messaging with reliable, traceable communication |
| `ObservableVectorDB` | Long-term memory system providing contextual information to agents |

---

## Project Structure

```
Multi-Agent-Reinforcement-Learning-for-Dynamic-Pricing/
│
├── main_execution.py          # Main entry point for running the system
├── config_settings.py         # Configuration and API key management
├── data_models.py             # Pydantic models and data structures
├── data_generator.py          # Synthetic data generation utilities
│
├── llm_agent.py               # Base and specialized agent implementations
├── market_environment.py      # Market simulation environment
├── memory_system.py           # Agent memory and state management
├── rag_system.py              # RAG system with vector database integration
│
├── simulator.py               # Simulation runner and orchestration
├── visualization.py           # Plotting and visualization utilities
│
├── Retail_AI_Agent_Army.ipynb # Interactive Jupyter notebook demo
└── README.md                  # This file
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- OpenAI API Key
- LangSmith API Key (optional but recommended for observability)

### Step 1: Clone the Repository

```bash
git clone https://github.com/jupadhya1/Multi-Agent-Reinforcement-Learning-for-Dynamic-Pricing.git
cd Multi-Agent-Reinforcement-Learning-for-Dynamic-Pricing
```

### Step 2: Install Dependencies

```bash
pip install langchain langchain-openai langsmith chromadb sentence-transformers pandas numpy matplotlib
```

Or let the script auto-install dependencies on first run.

### Step 3: Configure API Keys

**Option A: Environment Variables (Recommended)**

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="retail-agent-army"
```

**Option B: Direct Configuration**

Edit `config_settings.py` and replace placeholder values:

```python
OPENAI_API_KEY = "your-openai-api-key"
LANGSMITH_API_KEY = "your-langsmith-api-key"
LANGSMITH_PROJECT = "retail-agent-army"
```

---

## Usage

### Running the Demo

```bash
python main_execution.py
```

The script will:

1. Install any missing packages automatically
2. Initialize all system components (MCP Server, A2A Layer, Agents, etc.)
3. Run the observable pricing workflow for sample products
4. Print comprehensive observability data in JSON format

### Using the Jupyter Notebook

For an interactive experience:

```bash
jupyter notebook Retail_AI_Agent_Army.ipynb
```

---

## Observability

### Console Output

The script outputs a JSON object containing performance metrics for all components, providing a quick text-based summary of the entire run.

### LangSmith Dashboard (Recommended)

1. Log in to your [LangSmith account](https://smith.langchain.com/)
2. Navigate to the `retail-agent-army` project
3. Explore detailed, interactive traces for each workflow

**What you can inspect:**
- Every step of agent execution
- Inputs/outputs of all LLM calls
- Tool usage and performance
- Inter-agent communication patterns
- End-to-end latency breakdowns

---

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **Core Framework** | LangChain, LangChain OpenAI |
| **LLM** | OpenAI GPT-4 |
| **Observability** | LangSmith |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence-Transformers |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib |
| **Async Runtime** | Asyncio |

---

## Key Concepts

### Model Context Protocol (MCP)

MCP provides a standardized way for agents to access domain-specific tools. In this system, the simulated MCP server exposes:

- **`price_optimizer`**: Calculates optimal pricing based on market conditions
- **`demand_forecaster`**: Predicts demand based on historical data and trends

### Agent-to-Agent (A2A) Communication

The A2A layer enables agents to:
- Share insights discovered during analysis
- Broadcast important market signals
- Coordinate on complex multi-step decisions

### Observable Vector Database

The vector database serves as the system's knowledge base, storing:
- Business policies and rules
- Historical pricing data
- Market research and competitive intelligence

---

---

<p align="center">
  <sub>Built with ❤️ for the AI and Retail community</sub>
</p>
