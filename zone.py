"""
Module for representing geographic zones in the simulation.
Manages environmental parameters that affect citizen health outcomes.
"""

from typing import Dict, Optional


class Zone:
    """
    Represents a geographic zone in the urban area.
    
    Attributes:
        id: Unique identifier for the zone
        environmental_parameters: Dictionary of environmental characteristics
                                 (e.g., pollution, greenery, healthcare access)
    """
    
    # Class variable for unique ID generation
    _id_counter = 0
    
    def __init__(self, environmental_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize a zone.
        
        Args:
            environmental_parameters: Dictionary of environmental parameters.
                                     Default parameters include:
                                     - air_quality (0-1): 1 = clean, 0 = polluted
                                     - greenery_index (0-1): proportion of green space
                                     - healthcare_access (0-1): 1 = full access
                                     - population_density (people/km2): human density
        """
        Zone._id_counter += 1
        self.id: int = Zone._id_counter
        
        # Default environmental parameters
        self.environmental_parameters: Dict[str, float] = {
            "air_quality": 0.7,
            "greenery_index": 0.5,
            "healthcare_access": 0.8,
            "population_density": 5000.0,
        }
        
        # Override with provided parameters
        if environmental_parameters:
            self.environmental_parameters.update(environmental_parameters)
    
    def get_parameter(self, param_name: str, default: float = 0.5) -> float:
        """
        Get a specific environmental parameter.
        
        Args:
            param_name: Name of the parameter
            default: Default value if parameter not found
        
        Returns:
            Parameter value or default
        """
        return self.environmental_parameters.get(param_name, default)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Zone(id={self.id}, air_quality={self.environmental_parameters['air_quality']:.1f})"
