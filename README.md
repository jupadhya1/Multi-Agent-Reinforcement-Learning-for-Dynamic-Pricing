Retail AI Agent Army with Full Observability
This project demonstrates an advanced multi-agent system designed for complex retail decision-making, specifically focusing on dynamic pricing. It showcases an enterprise-grade architecture with a strong emphasis on observability, using LangSmith for end-to-end tracing and monitoring.



Shutterstock
🚀 Core Features
Multi-Agent System: Utilizes a team of specialized AI agents that collaborate to solve complex retail problems.

Model Context Protocol (MCP): A simulated MCP server provides agents with access to powerful, domain-specific tools like price_optimizer and demand_forecaster.

Agent-to-Agent (A2A) Communication: A dedicated communication layer allows agents to interact, share insights, and broadcast information across the network.

Observable Vector Database: Enriches agents' context with relevant business knowledge, policies, and historical data through a ChromaDB-powered vector store with full performance monitoring.

End-to-End Observability: Integrates seamlessly with LangSmith to provide detailed, interactive traces of every agent action, tool usage, and LLM call, making the system transparent and debuggable.

Automated Workflow Orchestration: A central orchestrator manages complex workflows, deploying the right agents for specific tasks and aggregating their findings into a cohesive strategy.

🏗️ System Architecture
The system is designed with a modular and observable architecture, where each component has a distinct responsibility and is tracked for performance.

ObservableRetailOrchestrator: The "brain" of the operation. It initiates workflows, delegates tasks to specialized agents, and aggregates their final recommendations.

EnhancedRetailAgent (Base Class): The blueprint for all agents, equipped with connections to the MCP, A2A layer, and Vector DB.

EnhancedPricingAgent (Specialized Agent): An agent specifically designed to handle pricing analysis. It uses its tools and knowledge to develop comprehensive pricing strategies.

RetailMCPServer: A collection of simulated, high-power tools that agents can invoke to perform complex calculations and analyses (e.g., price optimization, demand forecasting).

A2ACommunicationLayer: Manages all inter-agent messaging, ensuring reliable and traceable communication.

ObservableVectorDB: Acts as the system's long-term memory, providing agents with contextual information from a knowledge base. All queries are logged and monitored.

Workflow Diagram
[Orchestrator]
      |
      +--> [VectorDB] (Get Market Context)
      |
      +--> Deploys [Pricing Agent 1] & [Pricing Agent 2]
                  /        \
                 /          \
    (Uses)--> [MCP Tools]   [VectorDB] <-- (Uses)
                 \          /
                  \        /
      (Broadcasts via A2A Layer)
                   |
                   |
      +--> [Orchestrator] (Aggregates Results)
      |
[Final Strategy]

🛠️ Technology Stack
Python 3.10+

LangChain & LangChain OpenAI: Core frameworks for building with LLMs.

LangSmith: For AI observability and tracing.

OpenAI GPT-4: The primary model for agent reasoning and tool use.

ChromaDB: For the local vector database.

Sentence-Transformers: For generating text embeddings.

Pandas, NumPy, Matplotlib: For data manipulation and visualization.

Asyncio: For concurrent execution of agent tasks.

⚙️ Setup and Installation
Follow these steps to get the project running on your local machine.

1. Prerequisites
Python 3.10 or newer.

An active OpenAI API Key.

A LangSmith API Key (optional but highly recommended for observability).

2. Install Dependencies
The Python script is designed to be self-contained. When you run it for the first time, it will automatically check for and install all the required packages using pip.

3. Configure API Keys
Open the retail_agent_army.py file and navigate to Step 3. Replace the placeholder values with your actual API keys:

# IMPORTANT: Replace these with your actual keys.
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', "YOUR_OPENAI_API_KEY")
LANGSMITH_API_KEY = os.environ.get('LANGSMITH_API_KEY', "YOUR_LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.environ.get('LANGSMITH_PROJECT', "retail-agent-army")

For better security, it is recommended to set these as environment variables.

▶️ How to Run the Demo
With the API keys configured, run the script from your terminal:

python retail_agent_army.py

The script will perform the following actions:

Install any missing packages.

Initialize all system components (MCP Server, A2A Layer, Agents, etc.).

Run the observable pricing workflow for two sample products.

Print the final, comprehensive observability data to the console in JSON format.

📊 Checking the Observability Dashboard
There are two ways to view the results:

Console JSON Output: The script prints a final JSON object containing performance metrics for all components. This gives you a quick, text-based summary of the entire run.

LangSmith Web Interface (Recommended):

Log in to your LangSmith account.

Navigate to the retail-agent-army project.

You will find detailed, interactive traces for each workflow. You can inspect every step, view the inputs/outputs of LLM calls, and analyze the performance of each agent and tool. This is the best way to debug and understand the system's behavior.

This project serves as a robust template for building enterprise-ready multi-agent systems that are not only powerful but also transparent, manageable, and fully observable.
