"""
Module for representing households in the simulation.
Manages groups of citizens living together.
"""

from typing import List


class Household:
    """
    Represents a household containing multiple citizens.
    
    Attributes:
        id: Unique identifier for the household
        zone_id: Geographic zone identifier
        members: List of citizen IDs in this household
    """
    
    # Class variable for unique ID generation
    _id_counter = 0
    
    def __init__(self, zone_id: int) -> None:
        """
        Initialize a household.
        
        Args:
            zone_id: Geographic zone identifier
        """
        Household._id_counter += 1
        self.id: int = Household._id_counter
        self.zone_id: int = zone_id
        self.members: List[int] = []
    
    def add_member(self, citizen_id: int) -> None:
        """
        Add a citizen to the household.
        
        Args:
            citizen_id: ID of the citizen to add
        """
        if citizen_id not in self.members:
            self.members.append(citizen_id)
    
    def remove_member(self, citizen_id: int) -> bool:
        """
        Remove a citizen from the household.
        
        Args:
            citizen_id: ID of the citizen to remove
        
        Returns:
            True if citizen was removed, False if not in household
        """
        if citizen_id in self.members:
            self.members.remove(citizen_id)
            return True
        return False
    
    def size(self) -> int:
        """Get the number of members in the household."""
        return len(self.members)
    
    def has_children(self) -> bool:
        """Check if household contains any children (simplified check)."""
        # This would need access to citizen data, so we keep it simple
        # In actual usage, we'd check citizen ages through the engine
        return self.size() > 0
    
    def is_empty(self) -> bool:
        """Check if household is empty."""
        return self.size() == 0
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Household(id={self.id}, zone={self.zone_id}, members={self.size()})"
