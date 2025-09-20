# models/data_models.py
"""
Pydantic data models for structured data validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    """Agent behavioral types"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    COLLABORATIVE = "collaborative"

class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class MessageType(str, Enum):
    """Communication message types"""
    STRATEGY_SHARE = "strategy_share"
    MARKET_INFO = "market_info"
    COORDINATION = "coordination"
    WARNING = "warning"

class Priority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class MarketAnalysis(BaseModel):
    """Structured market analysis from LLM agents"""
    market_trend: str = Field(description="Overall market trend direction")
    competition_level: float = Field(
        description="Competition intensity (0-1)",
        ge=0.0, le=1.0
    )
    demand_forecast: str = Field(description="Expected demand direction")
    price_sensitivity: float = Field(
        description="Customer price sensitivity (0-1)",
        ge=0.0, le=1.0
    )
    recommended_strategy: str = Field(description="Recommended pricing strategy")
    confidence_score: float = Field(
        description="Confidence in analysis (0-1)",
        ge=0.0, le=1.0
    )
    key_insights: List[str] = Field(description="Key market insights")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('key_insights')
    def validate_insights(cls, v):
        if len(v) > 10:
            raise ValueError("Too many insights, maximum 10 allowed")
        return v

class PricingDecision(BaseModel):
    """Structured pricing decision from LLM agent"""
    product_id: str = Field(description="Product identifier")
    current_price: float = Field(description="Current product price", gt=0)
    recommended_price: float = Field(description="Recommended new price", gt=0)
    price_change_pct: float = Field(description="Percentage price change")
    reasoning: str = Field(description="Detailed reasoning for price change")
    risk_assessment: RiskLevel = Field(description="Risk level assessment")
    expected_impact: str = Field(description="Expected market impact")
    confidence: float = Field(
        description="Confidence in decision (0-1)",
        ge=0.0, le=1.0
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('price_change_pct', pre=True, always=True)
    def calculate_price_change(cls, v, values):
        if 'current_price' in values and 'recommended_price' in values:
            current = values['current_price']
            recommended = values['recommended_price']
            return ((recommended - current) / current) * 100
        return v

class AgentCommunication(BaseModel):
    """Message structure for inter-agent communication"""
    sender_id: str = Field(description="ID of sending agent")
    receiver_id: str = Field(description="ID of receiving agent or 'broadcast'")
    message_type: MessageType = Field(description="Type of message")
    content: Dict[str, Any] = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: Priority = Field(description="Message priority", default=Priority.MEDIUM)
    
    class Config:
        use_enum_values = True

class PerformanceMetrics(BaseModel):
    """Agent performance metrics"""
    revenue: float = Field(description="Total revenue generated", ge=0)
    market_share: float = Field(
        description="Market share percentage",
        ge=0.0, le=100.0
    )
    price_stability: float = Field(
        description="Price volatility measure",
        ge=0.0, le=1.0
    )
    customer_satisfaction: float = Field(
        description="Estimated satisfaction",
        ge=0.0, le=1.0
    )
    competitive_position: str = Field(description="Position vs competitors")
    strategic_effectiveness: float = Field(
        description="Strategy success rate",
        ge=0.0, le=1.0
    )
    timestamp: datetime = Field(default_factory=datetime.now)

class ProductData(BaseModel):
    """Product information model"""
    product_id: str = Field(description="Unique product identifier")
    category: str = Field(description="Product category")
    base_price: float = Field(description="Base product price", gt=0)
    current_price: float = Field(description="Current market price", gt=0)
    cost: float = Field(description="Product cost", gt=0)
    price_elasticity: float = Field(
        description="Price elasticity coefficient",
        lt=0  # Elasticity should be negative
    )
    seasonal_factor: float = Field(
        description="Seasonal adjustment factor",
        gt=0
    )
    brand_strength: float = Field(
        description="Brand strength score",
        ge=1.0, le=10.0
    )
    quality_rating: float = Field(
        description="Quality rating",
        ge=1.0, le=5.0
    )
    description: str = Field(description="Product description")
    
    @validator('current_price', pre=True, always=True)
    def set_current_price(cls, v, values):
        if v is None and 'base_price' in values:
            return values['base_price']
        return v

class MarketConditions(BaseModel):
    """Market environment conditions"""
    competition_level: float = Field(
        description="Market competition intensity",
        ge=0.0, le=1.0
    )
    demand_volatility: float = Field(
        description="Demand volatility measure",
        ge=0.0, le=1.0
    )
    seasonal_factor: float = Field(
        description="Seasonal adjustment",
        gt=0
    )
    economic_indicator: float = Field(
        description="Economic health indicator",
        ge=0.0, le=1.0
    )
    innovation_rate: float = Field(
        description="Market innovation rate",
        ge=0.0, le=1.0,
        default=0.4
    )
    week: int = Field(description="Simulation week", ge=1)

class SimulationWeekResult(BaseModel):
    """Results from one week of simulation"""
    week: int = Field(description="Week number", ge=1)
    market_conditions: MarketConditions = Field(description="Week's market state")
    agent_revenues: Dict[str, float] = Field(description="Revenue by agent")
    total_revenue: float = Field(description="Total market revenue", ge=0)
    decisions_made: int = Field(description="Total decisions made", ge=0)
    communications_sent: int = Field(description="Messages sent", ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)

class SimulationSummary(BaseModel):
    """Complete simulation results summary"""
    total_weeks: int = Field(description="Simulation duration", gt=0)
    total_revenue: float = Field(description="Total market revenue", ge=0)
    avg_weekly_revenue: float = Field(description="Average weekly revenue", ge=0)
    total_decisions: int = Field(description="Total decisions made", ge=0)
    total_communications: int = Field(description="Total messages sent", ge=0)
    decisions_per_week: float = Field(description="Average decisions per week", ge=0)
    communications_per_week: float = Field(description="Average messages per week", ge=0)
    start_time: datetime = Field(description="Simulation start time")
    end_time: datetime = Field(description="Simulation end time")

class AgentPerformanceSummary(BaseModel):
    """Agent performance over entire simulation"""
    agent_id: str = Field(description="Agent identifier")
    agent_type: AgentType = Field(description="Agent behavioral type")
    total_revenue: float = Field(description="Total revenue generated", ge=0)
    avg_weekly_revenue: float = Field(description="Average weekly revenue", ge=0)
    final_market_share: float = Field(
        description="Final market share percentage",
        ge=0.0, le=100.0
    )
    revenue_stability: float = Field(
        description="Revenue consistency measure",
        ge=0.0, le=1.0
    )
    decisions_made: int = Field(description="Total decisions made", ge=0)
    successful_predictions: int = Field(description="Successful predictions", ge=0)
    communication_activity: int = Field(description="Messages sent", ge=0)
    
    class Config:
        use_enum_values = True
