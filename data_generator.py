# simulation/data_generator.py
"""
Synthetic data generation for product catalog and market simulation
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from config.settings import config

logger = logging.getLogger(__name__)

@dataclass
class CategoryParams:
    """Parameters for product category generation"""
    name: str
    price_mean: float
    price_std: float
    elasticity_range: tuple
    quality_range: tuple
    brand_range: tuple
    seasonal_factor_range: tuple

class ProductDataGenerator:
    """Generate realistic synthetic product data for simulation"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define category parameters based on real market characteristics
        self.category_params = {
            'Electronics': CategoryParams(
                name='Electronics',
                price_mean=5.5,  # log-normal mean for ~$245 average
                price_std=0.8,
                elasticity_range=(-1.2, -0.6),
                quality_range=(3.5, 4.8),
                brand_range=(6.0, 9.5),
                seasonal_factor_range=(0.9, 1.4)
            ),
            'Clothing': CategoryParams(
                name='Clothing',
                price_mean=4.2,  # ~$67 average
                price_std=0.6,
                elasticity_range=(-1.5, -0.8),
                quality_range=(2.5, 4.5),
                brand_range=(4.0, 8.5),
                seasonal_factor_range=(0.8, 1.3)
            ),
            'Home & Garden': CategoryParams(
                name='Home & Garden',
                price_mean=4.8,  # ~$121 average
                price_std=0.7,
                elasticity_range=(-1.0, -0.5),
                quality_range=(3.0, 4.5),
                brand_range=(5.0, 8.0),
                seasonal_factor_range=(0.9, 1.2)
            ),
            'Sports': CategoryParams(
                name='Sports',
                price_mean=4.5,  # ~$90 average
                price_std=0.6,
                elasticity_range=(-1.3, -0.7),
                quality_range=(3.0, 4.8),
                brand_range=(5.5, 9.0),
                seasonal_factor_range=(0.8, 1.4)
            ),
            'Books': CategoryParams(
                name='Books',
                price_mean=3.2,  # ~$25 average
                price_std=0.4,
                elasticity_range=(-1.8, -1.0),
                quality_range=(3.5, 4.9),
                brand_range=(3.0, 7.0),
                seasonal_factor_range=(0.9, 1.1)
            ),
            'Toys': CategoryParams(
                name='Toys',
                price_mean=3.8,  # ~$45 average
                price_std=0.5,
                elasticity_range=(-1.6, -0.9),
                quality_range=(2.8, 4.2),
                brand_range=(4.0, 8.5),
                seasonal_factor_range=(0.7, 1.6)
            ),
            'Health': CategoryParams(
                name='Health',
                price_mean=4.0,  # ~$55 average
                price_std=0.6,
                elasticity_range=(-0.8, -0.4),
                quality_range=(3.8, 4.9),
                brand_range=(6.0, 9.0),
                seasonal_factor_range=(0.9, 1.1)
            ),
            'Automotive': CategoryParams(
                name='Automotive',
                price_mean=5.0,  # ~$148 average
                price_std=0.8,
                elasticity_range=(-1.1, -0.5),
                quality_range=(3.2, 4.6),
                brand_range=(5.5, 9.5),
                seasonal_factor_range=(0.9, 1.2)
            )
        }
    
    def generate_products(self, n_products: int = None) -> pd.DataFrame:
        """Generate enhanced synthetic product data"""
        n_products = n_products or config.simulation.total_products
        
        logger.info(f"Generating {n_products} synthetic products...")
        
        products = []
        categories = list(self.category_params.keys())
        
        for i in range(n_products):
            # Select category with some distribution
            category = np.random.choice(categories)
            params = self.category_params[category]
            
            # Generate base price using log-normal distribution
            base_price = np.random.lognormal(params.price_mean, params.price_std)
            
            # Calculate cost (typically 40-70% of price)
            cost_ratio = np.random.uniform(0.4, 0.7)
            cost = base_price * cost_ratio
            
            # Generate other attributes
            price_elasticity = np.random.uniform(*params.elasticity_range)
            seasonal_factor = np.random.uniform(*params.seasonal_factor_range)
            brand_strength = np.random.uniform(*params.brand_range)
            quality_rating = np.random.uniform(*params.quality_range)
            
            # Create product description
            product_number = (i % 100) + 1
            description = f"{category} Product #{product_number:03d}"
            
            product = {
                'product_id': f'PROD_{i:06d}',
                'category': category,
                'base_price': round(base_price, 2),
                'current_price': round(base_price, 2),  # Initialize to base price
                'cost': round(cost, 2),
                'price_elasticity': round(price_elasticity, 3),
                'seasonal_factor': round(seasonal_factor, 3),
                'brand_strength': round(brand_strength, 2),
                'quality_rating': round(quality_rating, 2),
                'description': description
            }
            
            products.append(product)
        
        df = pd.DataFrame(products)
        
        # Add some realistic correlations
        df = self._add_correlations(df)
        
        # Validate data
        df = self._validate_product_data(df)
        
        logger.info(f"Generated {len(df)} products across {df['category'].nunique()} categories")
        self._log_data_summary(df)
        
        return df
    
    def _add_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic correlations between product attributes"""
        try:
            # Higher quality products tend to have higher brand strength
            quality_boost = (df['quality_rating'] - 3.0) * 0.5
            df['brand_strength'] = np.clip(
                df['brand_strength'] + quality_boost, 
                1.0, 10.0
            )
            
            # Premium products (high price) tend to be less elastic
            price_percentile = df['base_price'].rank(pct=True)
            elasticity_adjustment = (price_percentile - 0.5) * 0.3
            df['price_elasticity'] = np.clip(
                df['price_elasticity'] + elasticity_adjustment,
                -2.0, -0.2
            )
            
            # Round adjusted values
            df['brand_strength'] = df['brand_strength'].round(2)
            df['price_elasticity'] = df['price_elasticity'].round(3)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding correlations: {e}")
            return df
    
    def _validate_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean product data"""
        try:
            # Remove any invalid products
            initial_count = len(df)
            
            # Ensure positive prices and costs
            df = df[df['base_price'] > 0]
            df = df[df['cost'] > 0]
            df = df[df['cost'] < df['base_price']]  # Cost should be less than price
            
            # Ensure valid ranges
            df = df[df['quality_rating'].between(1.0, 5.0)]
            df = df[df['brand_strength'].between(1.0, 10.0)]
            df = df[df['price_elasticity'] < 0]  # Should be negative
            
            # Reset index if any rows were removed
            if len(df) < initial_count:
                df = df.reset_index(drop=True)
                logger.warning(f"Removed {initial_count - len(df)} invalid products")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating product data: {e}")
            return df
    
    def _log_data_summary(self, df: pd.DataFrame):
        """Log summary statistics of generated data"""
        try:
            logger.info("Product Data Summary:")
            logger.info(f"  Categories: {df['category'].value_counts().to_dict()}")
            logger.info(f"  Price range: ${df['base_price'].min():.2f} - ${df['base_price'].max():.2f}")
            logger.info(f"  Average price: ${df['base_price'].mean():.2f}")
            logger.info(f"  Average elasticity: {df['price_elasticity'].mean():.3f}")
            logger.info(f"  Quality range: {df['quality_rating'].min():.1f} - {df['quality_rating'].max():.1f}")
            
        except Exception as e:
            logger.error(f"Error logging data summary: {e}")

class MarketConditionGenerator:
    """Generate realistic market condition scenarios"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_scenario(self, scenario_type: str = "normal") -> Dict[str, Any]:
        """Generate market condition scenario"""
        scenarios = {
            'normal': {
                'competition_level': np.random.uniform(0.4, 0.6),
                'demand_volatility': np.random.uniform(0.2, 0.4),
                'economic_indicator': np.random.uniform(0.5, 0.7),
                'description': 'Normal market conditions'
            },
            'high_competition': {
                'competition_level': np.random.uniform(0.7, 0.9),
                'demand_volatility': np.random.uniform(0.3, 0.5),
                'economic_indicator': np.random.uniform(0.4, 0.6),
                'description': 'High competition market'
            },
            'economic_downturn': {
                'competition_level': np.random.uniform(0.6, 0.8),
                'demand_volatility': np.random.uniform(0.4, 0.7),
                'economic_indicator': np.random.uniform(0.2, 0.4),
                'description': 'Economic downturn scenario'
            },
            'boom_market': {
                'competition_level': np.random.uniform(0.3, 0.5),
                'demand_volatility': np.random.uniform(0.1, 0.3),
                'economic_indicator': np.random.uniform(0.7, 0.9),
                'description': 'Market boom scenario'
            },
            'volatile': {
                'competition_level': np.random.uniform(0.4, 0.7),
                'demand_volatility': np.random.uniform(0.6, 0.9),
                'economic_indicator': np.random.uniform(0.3, 0.7),
                'description': 'High volatility market'
            }
        }
        
        if scenario_type not in scenarios:
            logger.warning(f"Unknown scenario type: {scenario_type}, using normal")
            scenario_type = 'normal'
        
        scenario = scenarios[scenario_type].copy()
        scenario['scenario_type'] = scenario_type
        scenario['seasonal_factor'] = 1.0  # Will be modified during simulation
        scenario['innovation_rate'] = np.random.uniform(0.3, 0.6)
        
        return scenario

def generate_agent_configurations(agent_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Generate standard agent configurations"""
    if agent_types is None:
        agent_types = ['adaptive', 'aggressive', 'conservative', 'collaborative']
    
    configurations = []
    
    agent_type_mapping = {
        'adaptive': 'MADDPG_Agent',
        'aggressive': 'MADQN_Agent', 
        'conservative': 'Conservative_Agent',
        'collaborative': 'QMIX_Agent'
    }
    
    personality_mapping = {
        'adaptive': 'balanced',
        'aggressive': 'competitive',
        'conservative': 'risk_averse',
        'collaborative': 'cooperative'
    }
    
    for agent_type in agent_types:
        if agent_type in agent_type_mapping:
            config_item = {
                'agent_id': agent_type_mapping[agent_type],
                'agent_type': agent_type,
                'personality': personality_mapping[agent_type]
            }
            configurations.append(config_item)
        else:
            logger.warning(f"Unknown agent type: {agent_type}")
    
    return configurations

def create_sample_datasets() -> Dict[str, pd.DataFrame]:
    """Create sample datasets for testing and demonstration"""
    generator = ProductDataGenerator()
    
    datasets = {
        'small': generator.generate_products(20),
        'medium': generator.generate_products(50),
        'large': generator.generate_products(100),
        'xlarge': generator.generate_products(200)
    }
    
    logger.info(f"Created {len(datasets)} sample datasets")
    return datasets

# Convenience functions for quick data generation
def generate_quick_dataset(size: str = "medium") -> pd.DataFrame:
    """Generate a quick dataset for immediate use"""
    size_mapping = {
        'small': 20,
        'medium': 50,
        'large': 100,
        'xlarge': 200
    }
    
    n_products = size_mapping.get(size, 50)
    generator = ProductDataGenerator()
    return generator.generate_products(n_products)

def get_default_agent_configs() -> List[Dict[str, Any]]:
    """Get default agent configurations matching the research paper"""
    return generate_agent_configurations()
