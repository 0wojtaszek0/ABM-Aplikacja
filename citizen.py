"""
Module for representing individual citizens in the simulation.
Handles aging, disease progression, mortality risk, and fertility.
"""

from typing import Dict, Optional
import random
import math
from disease_model import DiseaseModel


class Citizen:
    """
    Represents an individual citizen in the population.
    
    Attributes:
        id: Unique identifier for the citizen
        sex: Biological sex ("male" or "female")
        age_months: Age in months
        alive: Whether the citizen is still alive
        household_id: ID of the household the citizen belongs to
        zone_id: ID of the zone the citizen lives in
        diseases: Dictionary of disease names and their stage (0/1)
        risk_factors: Dictionary of risk factors and their presence (0/1)
        disability_score: Computed disability score based on diseases
        health_state: Current health state of the citizen
        rng: Random number generator
    """
    
    # Class variable for unique ID generation
    _id_counter = 0
    
    # Default risk factors
    DEFAULT_RISK_FACTORS = [
        "smoking",
        "obesity",
        "physical_inactivity",
        "alcohol_abuse",
        "high_cholesterol",
        "hypertension_stage0",
        "family_history",
    ]
    
    def __init__(
        self,
        sex: str,
        age_months: int,
        household_id: int,
        zone_id: int = 1,
        diseases: Optional[Dict[str, int]] = None,
        risk_factors: Optional[Dict[str, int]] = None,
        health_state: str = "healthy",
    ) -> None:
        """
        Initialize a citizen.
        
        Args:
            sex: "male" or "female"
            age_months: Age in months
            household_id: ID of household
            zone_id: ID of zone
            diseases: Dictionary of diseases and their presence (0/1)
            risk_factors: Dictionary of risk factors and their presence (0/1)
            health_state: Current health state of the citizen
        """
        Citizen._id_counter += 1
        self.id: int = Citizen._id_counter
        self.sex: str = sex
        self.age_months: int = age_months
        self.alive: bool = True
        self.household_id: int = household_id
        self.zone_id: int = zone_id
        self.diseases: Dict[str, int] = diseases or {}
        self.risk_factors: Dict[str, int] = risk_factors or {rf: 0 for rf in self.DEFAULT_RISK_FACTORS}
        self.disability_score: float = 0.0
        self.health_state: str = health_state
        self.rng: random.Random = random.Random()
        # Skumulowany hazard biologiczny per choroba (Cox-style memory).
        # Inicjalizowany do 0 dla każdej znanej choroby; uzupełniany leniwie w handle_deaths.
        self.cumulative_hazard: Dict[str, float] = {d: 0.0 for d in self.diseases.keys()}
        self.compute_disability_score()
    
    def age_one_month(self) -> None:
        """Increment age by one month."""
        self.age_months += 1
    
    @property
    def age_years(self) -> float:
        """Get age in years."""
        return self.age_months / 12.0
    
    def compute_disability_score(self, disease_weights: Optional[Dict[str, float]] = None) -> None:
        """
        Compute disability score as sum of disease weights.
        
        Args:
            disease_weights: Dictionary mapping disease names to disability weights.
                           If None, each disease counts as 0.1.
        """
        if disease_weights is None:
            disease_weights = {disease: 0.1 for disease in self.diseases.keys()}
        
        self.disability_score = sum(
            disease_weights.get(disease, 0.1) 
            for disease, presence in self.diseases.items() 
            if presence == 1
        )
    
    def num_conditions(self) -> int:
        """Get number of diseases the citizen has."""
        return sum(self.diseases.values())
    
    def has_multimorbidity(self) -> bool:
        """Check if citizen has multimorbidity (>= 2 conditions)."""
        return self.num_conditions() >= 2
    
    def mortality_risk(
        self,
        base_mortality_rate: float = 0.00005,
        mortality_multiplier: float = 1.0,
        female_advantage: float = 0.85,
    ) -> float:
        """
        Calculate monthly mortality risk based on age, conditions, disability, and risk factors.
        
        Risk model:
        - Base risk increases with age (exponential)
        - Increases with number of conditions
        - Increases with disability score
        - Increases with number of risk factors
        - Females have 15% lower risk
        
        Args:
            base_mortality_rate: Base monthly mortality rate
            mortality_multiplier: Multiplier for disease/age effects
            female_advantage: Mortality reduction for females (0-1)
        
        Returns:
            Probability of death in current month (0-1)
        """
        age_years = self.age_years
        
        # Age-related risk (exponential increase after 50)
        age_factor = base_mortality_rate * (1.0 + math.exp((age_years - 55) / 15))
        
        # Disease and disability contribution
        disease_factor = mortality_multiplier * (
            0.001 * self.num_conditions() + 
            0.005 * self.disability_score
        )
        
        # Risk factor contribution
        risk_factor_count = sum(self.risk_factors.values())
        risk_factor_contribution = 0.0005 * risk_factor_count
        
        total_risk = age_factor + disease_factor + risk_factor_contribution
        
        # Apply sex-based adjustment
        if self.sex == "female":
            total_risk *= female_advantage
        
        # Clamp to valid probability range
        return min(max(total_risk, 0.0), 1.0)
    
    def fertility_probability(self) -> float:
        """
        Calculate probability of giving birth this month.
        
        Only applicable to females aged 18-40.
        Peak fertility 25-30 years.
        Affected by disease burden.
        
        Returns:
            Probability of birth (0-1)
        """
        if self.sex != "female":
            return 0.0
        
        age_years = self.age_years
        
        # Only fertile between 18 and 40
        if age_years < 18 or age_years > 40:
            return 0.0
        
        # Peak fertility 25-30, gaussian distribution
        peak_age = 27.5
        fertility_base = math.exp(-((age_years - peak_age) ** 2) / 25)
        
        # Monthly fertility rate (adjusted for monthly scale)
        monthly_rate = fertility_base * 0.003  # ~3.6% annual rate at peak
        
        # Reduce fertility based on disease burden
        disease_reduction = 1.0 - (0.05 * self.num_conditions() + 0.1 * self.disability_score)
        disease_reduction = max(disease_reduction, 0.0)
        
        return monthly_rate * disease_reduction
    
    def maybe_die(self, rng: Optional[random.Random] = None) -> bool:
        """
        Determine if citizen dies this month based on mortality risk.
        
        Args:
            rng: Random number generator (defaults to global random)
        
        Returns:
            True if citizen dies, False otherwise
        """
        if not self.alive:
            return False
        
        if rng is None:
            rng = random.Random()
        
        risk = self.mortality_risk()
        if rng.random() < risk:
            self.alive = False
            return True
        return False
    
    def maybe_give_birth(self, rng: Optional[random.Random] = None) -> bool:
        """
        Determine if female citizen gives birth this month.
        
        Args:
            rng: Random number generator (defaults to global random)
        
        Returns:
            True if birth occurs, False otherwise
        """
        if not self.alive or self.sex != "female":
            return False
        
        if rng is None:
            rng = random.Random()
        
        prob = self.fertility_probability()
        return rng.random() < prob
    
    def update_health_state(self, disease_model: 'DiseaseModel') -> None:
        """
        Update the health state of the citizen based on transition probabilities.

        Args:
            disease_model: Instance of DiseaseModel to get transition probabilities
        """
        if not self.alive:
            return

        current_state = self.health_state
        for next_state in DiseaseModel.HEALTH_STATES:
            if current_state == next_state:
                continue

            transition_prob = disease_model.get_transition_probability(current_state, next_state)
            if self.rng.random() < transition_prob:
                self.health_state = next_state

                # If the citizen transitions to "severe" or "death", update alive status
                if self.health_state == "death":
                    self.alive = False
                break
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        age_str = f"{self.age_years:.1f}y"
        cond_str = f"{self.num_conditions()}c" if self.num_conditions() > 0 else "0c"
        status = "alive" if self.alive else "dead"
        return f"Citizen(id={self.id}, {self.sex[0]}, {age_str}, {cond_str}, {status})"
