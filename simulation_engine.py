"""
Main simulation engine for the Agent-Based Model.
Orchestrates the simulation loop and handles all demographic events.
Implements realistic Polish demographic structure and synthetic population generation.
"""

from typing import Dict, List, Tuple, Optional
import math
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import random
import pandas as pd

from citizen import Citizen
from household import Household
from zone import Zone
from disease_model import DiseaseModel


class SimulationEngine:
    """
    Main simulation engine managing population dynamics.
    
    Implements realistic Polish demographic structure based on GUS (Statistics Poland) data.
    
    Attributes:
        citizens: Dictionary mapping citizen IDs to Citizen objects
        households: Dictionary mapping household IDs to Household objects
        zones: Dictionary mapping zone IDs to Zone objects
        disease_model: DiseaseModel instance
        current_month: Current month in simulation
        yearly_stats: Dictionary storing statistics for each year
        rng: Random number generator for reproducibility
        mortality_table: Age-sex specific mortality rates (per month)
        fertility_table: Age-specific fertility rates (per year)
    """
    
    # Age-specific mortality rates (monthly) - Polish demographic data inspired
    # Higher values for males and elderly
    DEFAULT_MORTALITY_TABLE = {
        # Monthly death probability (male, female) — GUS Poland 2021 Tablice Trwania Życia
        # Males have substantially higher mortality at all ages (excess peaks at 20-65)
        0:  (0.000373, 0.000291),  # Infant — 4.47/3.49 per 1000/yr
        1:  (0.000015, 0.000012),  # Ages 1–4  — 0.18/0.14 /yr
        5:  (0.000010, 0.000008),  # Ages 5–9
        10: (0.000010, 0.000008),  # Ages 10–14
        15: (0.000018, 0.000009),  # Ages 15–19 — 0.22/0.11 /yr
        20: (0.000042, 0.000012),  # Ages 20–24 — 0.50/0.14 /yr (excess male: accidents)
        25: (0.000048, 0.000015),  # Ages 25–29 — 0.58/0.18 /yr
        30: (0.000060, 0.000021),  # Ages 30–34 — 0.72/0.25 /yr
        35: (0.000085, 0.000033),  # Ages 35–39 — 1.02/0.40 /yr
        40: (0.000167, 0.000067),  # Ages 40–44 — 2.0/0.8 /yr
        45: (0.000292, 0.000125),  # Ages 45–49 — 3.5/1.5 /yr
        50: (0.000500, 0.000208),  # Ages 50–54 — 6.0/2.5 /yr
        55: (0.000833, 0.000358),  # Ages 55–59 — 10.0/4.3 /yr
        60: (0.001375, 0.000583),  # Ages 60–64 — 16.5/7.0 /yr
        65: (0.002100, 0.001125),  # Ages 65–69 — 25.2/13.5 /yr (male ≈ 1.9× female)
        70: (0.003292, 0.001833),  # Ages 70–74 — 39.5/22.0 /yr (male ≈ 1.8×)
        75: (0.005042, 0.003083),  # Ages 75–79 — 60.5/37.0 /yr (male ≈ 1.6×)
        80: (0.007658, 0.005375),  # Ages 80–84 — 91.9/64.5 /yr (male ≈ 1.4×)
        85: (0.012400, 0.009000),  # Ages 85–89 — 148.8/108.0 /yr
        90: (0.018333, 0.013750),  # Ages 90+   — 220/165 /yr
    }

    # Age-specific fertility rates (annual) — Polish GUS 2021, TFR ≈ 1.26
    DEFAULT_FERTILITY_TABLE = {
        15: 0.011,  # Ages 15–19
        20: 0.041,  # Ages 20–24
        25: 0.081,  # Ages 25–29 (peak)
        30: 0.082,  # Ages 30–34 (peak)
        35: 0.034,  # Ages 35–39
        40: 0.007,  # Ages 40–44
        45: 0.0003, # Ages 45–49
    }
    
    def __init__(
        self,
        disease_model: Optional[DiseaseModel] = None,
        seed: Optional[int] = None,
        mortality_table: Optional[Dict[int, Tuple[float, float]]] = None,
        fertility_table: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Initialize the simulation engine.
        
        Args:
            disease_model: DiseaseModel instance (creates default if None)
            seed: Random seed for reproducibility
            mortality_table: Age-specific mortality rates (defaults to Polish-inspired data)
            fertility_table: Age-specific fertility rates (defaults to Polish-inspired data)
        """
        self.disease_model: DiseaseModel = disease_model or DiseaseModel()
        self.citizens: Dict[int, Citizen] = {}
        self.households: Dict[int, Household] = {}
        self.zones: Dict[int, Zone] = {}
        self.current_month: int = 0
        self.yearly_stats: Dict[int, Dict] = {}
        self.rng: random.Random = random.Random(seed)
        
        # Demographic parameters
        self.mortality_table: Dict[int, Tuple[float, float]] = (
            mortality_table or self.DEFAULT_MORTALITY_TABLE.copy()
        )
        self.fertility_table: Dict[int, float] = (
            fertility_table or self.DEFAULT_FERTILITY_TABLE.copy()
        )
        
        # Simulation parameters (optimized via GridSearch)
        # NOTE: These values are automatically set by main.py based on GridSearch results
        self.fertility_rate: float = 1.0
        """Fertility rate multiplier (default=1.0, optimized=2.3). Multiplies age-specific fertility rates."""
        
        self.mortality_multiplier: float = 1.0
        """Mortality multiplier (default=1.0, optimized=0.5). Multiplies all mortality rates in the model."""
        
        self.household_split_probability: float = 0.001
        """Probability per month that adults (25+) leave household to form new one (0.1% monthly)."""

        self.parallel_workers: int = mp.cpu_count() or 1
        """Number of worker processes used for per-step simulation parallelism."""
        
        # Create default zones
        self._init_zones()
    
    def _init_zones(self) -> None:
        """Initialize default zones with environmental parameters."""
        zone_params = [
            {"air_quality": 0.75, "greenery_index": 0.6, "healthcare_access": 0.9, "population_density": 4000},
            {"air_quality": 0.65, "greenery_index": 0.4, "healthcare_access": 0.85, "population_density": 6000},
            {"air_quality": 0.70, "greenery_index": 0.5, "healthcare_access": 0.80, "population_density": 5000},
            {"air_quality": 0.55, "greenery_index": 0.3, "healthcare_access": 0.75, "population_density": 8000},
        ]
        
        for params in zone_params:
            zone = Zone(params)
            self.zones[zone.id] = zone
    
    def _get_mortality_rate(self, age_years: float, sex: str) -> float:
        """
        Get mortality rate for given age and sex using interpolation.
        
        Args:
            age_years: Age in years
            sex: "male" or "female"
        
        Returns:
            Monthly mortality probability (0-1)
        """
        age_int = int(age_years)

        # Use floor bracket: find highest table key ≤ age_int
        available_ages = sorted(self.mortality_table.keys())
        bracket = available_ages[0]
        for a in available_ages:
            if a <= age_int:
                bracket = a
            else:
                break
        male_rate, female_rate = self.mortality_table[bracket]

        rate = male_rate if sex == "male" else female_rate
        return min(max(rate, 0.0), 1.0)
    
    def _get_fertility_rate(self, age_years: float) -> float:
        """
        Get annual fertility rate for given female age.
        
        Args:
            age_years: Age in years
        
        Returns:
            Annual fertility rate
        """
        if age_years < 15 or age_years > 50:
            return 0.0
        
        age_int = int(age_years)
        
        # Find matching or nearest age
        if age_int in self.fertility_table:
            return self.fertility_table[age_int]
        
        # Interpolate between ages
        available_ages = sorted([a for a in self.fertility_table.keys() if a <= age_int])
        if not available_ages:
            return self.fertility_table[min(self.fertility_table.keys())]
        
        return self.fertility_table[available_ages[-1]]
    
    def load_initial_population(
        self,
        filepath: str = "population_data.xlsx",
        min_age: int = 20,
        max_age: int = 80,
    ) -> int:
        """
        Load initial population from Excel file or create synthetic if not found.
        
        The Excel file should have columns:
        - id, sex, age (in years), household_id, zone_id
        - Plus columns for diseases (0/1)
        
        Args:
            filepath: Path to Excel file (optional)
            min_age: Minimum age to include (years)
            max_age: Maximum age to include (years)
        
        Returns:
            Number of citizens loaded
        """
        try:
            df = pd.read_excel(filepath)
            print(f"Loaded from {filepath}: {len(df)} potential citizens")
        except Exception as e:
            print(f"Could not load from Excel ({e}). Creating synthetic population of 50,000 citizens.")
            return self._create_synthetic_population(50000)
        
        # Assume Excel has columns: id, sex, age, household_id, zone_id, and disease columns
        diseases_list = self.disease_model.diseases
        
        count = 0
        households_seen = set()
        
        for _, row in df.iterrows():
            # Skip invalid ages
            if pd.isna(row.get("age")):
                continue
            
            age = int(row["age"])
            if age < min_age or age > max_age:
                continue
            
            sex = str(row.get("sex", "")).lower()
            if sex not in ["male", "female", "m", "f"]:
                sex = self.rng.choice(["male", "female"])
            
            # Normalize sex
            if sex == "m":
                sex = "male"
            elif sex == "f":
                sex = "female"
            
            # Create citizen
            age_months = age * 12 + self.rng.randint(0, 11)
            
            # Get diseases
            diseases_dict = self.disease_model.get_initial_diseases()
            for disease in diseases_list:
                if disease in row and row[disease] == 1:
                    diseases_dict[disease] = 1
            
            household_id = int(row.get("household_id", 0)) or self.rng.randint(1, 1000)
            zone_id = int(row.get("zone_id", 0)) or self.rng.choice(list(self.zones.keys()))
            
            citizen = Citizen(
                sex=sex,
                age_months=age_months,
                household_id=household_id,
                zone_id=zone_id,
                diseases=diseases_dict,
            )
            
            citizen.compute_disability_score(
                self.disease_model.get_all_disability_weights()
            )
            
            self.citizens[citizen.id] = citizen
            count += 1
        
        # Create households for all citizens
        for citizen in self.citizens.values():
            if citizen.household_id not in self.households:
                zone_id = citizen.zone_id
                household = Household(zone_id)
                self.households[household.id] = household
                citizen.household_id = household.id
            self.households[citizen.household_id].add_member(citizen.id)
        
        return count
    
    def _create_synthetic_population(self, size: int = 50000) -> int:
        """
        Create a synthetic population with realistic Polish demographic structure.
        
        Includes:
        - Age distribution 0-90 years reflecting Polish population pyramid
        - Children (age 0-14) generated synthetically
        - Realistic sex ratio (51-52% female)
        - Age-appropriate disease prevalence
        - Risk factors based on age and other characteristics
        
        Args:
            size: Number of citizens to create (default 50,000)
        
        Returns:
            Number of citizens created
        """
        print(f"Creating synthetic realistic Polish population of {size} citizens...")
        
        # Polish age distribution — GUS 2021, values sum to ~1.0 (normalized below)
        age_distribution = {
            0:  0.094,  # 0-9:   9.4%
            10: 0.098,  # 10-19: 9.8%
            20: 0.104,  # 20-29: 10.4%
            30: 0.143,  # 30-39: 14.3%
            40: 0.135,  # 40-49: 13.5%
            50: 0.137,  # 50-59: 13.7%
            60: 0.128,  # 60-69: 12.8%
            70: 0.097,  # 70-79: 9.7%
            80: 0.051,  # 80-89: 5.1%
            90: 0.013,  # 90+:   1.3% (sim goes up to age 100; Poland 90+ ≈ 0.54%)
        }
        
        # Normalize to ensure sum = 1.0
        total = sum(age_distribution.values())
        age_distribution = {age: prop / total for age, prop in age_distribution.items()}
        
        # Realistic Polish sex ratio (more females overall, even more at older ages)
        sex_ratio_by_decade = {
            0: 0.51,   # 51% female 0-9
            10: 0.50,  # 50% female 10-19
            20: 0.51,  # 51% female 20-29
            30: 0.51,  # 51% female 30-39
            40: 0.51,  # 51% female 40-49
            50: 0.52,  # 52% female 50-59
            60: 0.53,  # 53% female 60-69
            70: 0.55,  # 55% female 70-79
            80: 0.60,  # 60% female 80-89
            90: 0.65,  # 65% female 90+
        }
        
        # Create zones if not already done
        if not self.zones:
            self._init_zones()
        
        # Create households (average household size ~2.5)
        num_households = int(size / 2.5)
        household_members = [0] * num_households
        
        count = 0
        
        # Generate population by age group
        for decade_start in sorted(age_distribution.keys()):
            proportion = age_distribution[decade_start]
            count_in_decade = int(size * proportion)
            
            sex_ratio = sex_ratio_by_decade.get(decade_start, 0.51)
            
            for _ in range(count_in_decade):
                # Random age within decade
                age_years = decade_start + self.rng.random() * 10
                age_months = int(age_years * 12)
                
                # Determine sex based on age-specific ratio
                sex = "female" if self.rng.random() < sex_ratio else "male"
                
                # Initialize diseases based on age and prevalence
                diseases_dict = self.disease_model.get_initial_diseases()
                
                # Children (0-14) have very low disease prevalence
                if age_years < 14:
                    # Only rare congenital conditions
                    for disease in self.disease_model.diseases:
                        diseases_dict[disease] = 0  # No diseases in children
                else:
                    # Age-appropriate disease prevalence
                    age_factor = max(0, (age_years - 20) / 50)  # Increases with age
                    
                    for disease in self.disease_model.diseases:
                        base_prevalence = self.disease_model.get_prevalence(disease) / 100.0
                        # Higher prevalence in older ages
                        adjusted_prevalence = base_prevalence * (0.1 + age_factor)
                        if self.rng.random() < adjusted_prevalence:
                            diseases_dict[disease] = 1
                
                # Assign to household (distribute roughly evenly)
                household_idx = count % num_households
                household_id = list(self.households.keys())[0] if self.households else None
                
                zone_id = self.rng.choice(list(self.zones.keys()))
                
                # Create citizen
                citizen = Citizen(
                    sex=sex,
                    age_months=age_months,
                    household_id=household_id or 1,
                    zone_id=zone_id,
                    diseases=diseases_dict,
                )
                
                # Initialize risk factors based on age and characteristics
                citizen.risk_factors = self._init_risk_factors(citizen)
                
                citizen.compute_disability_score(
                    self.disease_model.get_all_disability_weights()
                )
                
                self.citizens[citizen.id] = citizen
                count += 1
        
        # Create households and assign citizens
        zone_list = list(self.zones.keys())
        for hh_idx in range(num_households):
            zone_id = zone_list[hh_idx % len(zone_list)]
            household = Household(zone_id)
            self.households[household.id] = household
        
        # Assign citizens to households
        household_ids = list(self.households.keys())
        for idx, citizen in enumerate(self.citizens.values()):
            household_id = household_ids[idx % len(household_ids)]
            citizen.household_id = household_id
            self.households[household_id].add_member(citizen.id)
        
        print(f"  Created {count} citizens in {len(self.households)} households")
        print(f"  Created {len(self.zones)} zones with environmental parameters")
        
        return count
    
    def _init_risk_factors(self, citizen: Citizen) -> Dict[str, int]:
        """
        Initialize risk factors for a citizen based on age and characteristics.
        
        Args:
            citizen: Citizen object
        
        Returns:
            Dictionary of risk factors
        """
        risk_factors = {rf: 0 for rf in Citizen.DEFAULT_RISK_FACTORS}
        
        age_years = citizen.age_years
        
        # Children have no risk factors
        if age_years < 15:
            return risk_factors
        
        # Smoking prevalence increases with age, peaks at 40-60, then decreases
        smoking_prob = 0.0
        if 20 <= age_years <= 70:
            peak_age = 45
            smoking_prob = 0.25 * (1 - ((age_years - peak_age) ** 2) / (50 ** 2))
            smoking_prob = max(smoking_prob, 0.10)
        if self.rng.random() < smoking_prob:
            risk_factors["smoking"] = 1
        
        # Obesity prevalence increases with age
        obesity_prob = 0.15 + (age_years - 20) * 0.008 if age_years > 20 else 0.05
        obesity_prob = min(obesity_prob, 0.45)
        if self.rng.random() < obesity_prob:
            risk_factors["obesity"] = 1
        
        # Physical inactivity increases with age
        inactivity_prob = 0.2 + (age_years - 20) * 0.005 if age_years > 20 else 0.1
        if self.rng.random() < inactivity_prob:
            risk_factors["physical_inactivity"] = 1
        
        # Alcohol abuse
        alcohol_prob = 0.08 if 20 <= age_years <= 65 else 0.02
        if self.rng.random() < alcohol_prob:
            risk_factors["alcohol_abuse"] = 1
        
        # High cholesterol increases with age
        cholesterol_prob = (age_years - 20) * 0.006 if age_years > 20 else 0.01
        if self.rng.random() < min(cholesterol_prob, 0.4):
            risk_factors["high_cholesterol"] = 1
        
        # Pre-hypertension stage
        hypertension_prob = (age_years - 30) * 0.008 if age_years > 30 else 0.01
        if self.rng.random() < min(hypertension_prob, 0.35):
            risk_factors["hypertension_stage0"] = 1
        
        # Family history (constant, independent of age)
        if self.rng.random() < 0.15:
            risk_factors["family_history"] = 1
        
        return risk_factors
    
    def run(self, months: int = 600) -> None:
        """
        Run the simulation for a specified number of months.
        
        Args:
            months: Number of months to simulate
        """
        print(f"Starting simulation for {months} months ({months/12:.1f} years)")
        print(f"Initial population: {len(self.citizens)} citizens")

        for month in range(months):
            self.step()

            if (month + 1) % 12 == 0:
                year = (month + 1) // 12
                self.collect_yearly_stats(year)
                if year % 5 == 0 or year == 1:
                    active_households = self._count_active_households()
                    print(
                        f"Year {year}: Population={len([c for c in self.citizens.values() if c.alive])}, "
                        f"Households={active_households} active / {len(self.households)} total"
                    )

        active_households = self._count_active_households()
        print(
            f"Simulation complete. Final population: {len([c for c in self.citizens.values() if c.alive])}, "
            f"Active households={active_households}, Total household objects={len(self.households)}"
        )
    
    def step(self) -> None:
        """
        Perform a single simulation step (one month).
        """
        citizens = list(self.citizens.values())
        worker_count = min(self.parallel_workers, max(1, len(citizens)))

        if worker_count > 1 and len(citizens) > worker_count * 2:
            chunk_size = max(1, len(citizens) // worker_count)
            chunks = self._chunked_list(citizens, chunk_size)
            with ThreadPool(worker_count) as pool:
                pool.map(self._age_citizen_batch, chunks)
        else:
            for citizen in citizens:
                if citizen.alive:
                    citizen.age_one_month()

        self.handle_deaths()
        self.handle_births()
        self.handle_household_splits()

        alive_citizens = [c for c in self.citizens.values() if c.alive]
        if worker_count > 1 and len(alive_citizens) > worker_count * 2:
            chunk_size = max(1, len(alive_citizens) // worker_count)
            chunks = self._chunked_list(alive_citizens, chunk_size)
            with ThreadPool(worker_count) as pool:
                pool.map(self._update_health_batch, chunks)
        else:
            for citizen in alive_citizens:
                citizen.update_health_state(self.disease_model)
    
    def handle_deaths(self) -> None:
        """
        Proces zgonów z dynamicznym modelem hazardu (Cox-style).

        Każdy żyjący agent przechodzi miesięcznie przez 3 fazy:

        1. **Akumulacja hazardu** — dla każdej z 3 chorób (CVD, Lung Cancer,
           Hypercholesterolemia) liczony jest miesięczny przyrost
           Δh = λ_0 · exp(γ·(age-30)) · exp(Σ β_i · RF_i), który dopisuje
           się do `citizen.cumulative_hazard[disease]` (biologiczna pamięć).

        2. **Onset choroby** — jeśli choroba nieaktywna, próba inicjacji
           z prawdopodobieństwem P = 1 - exp(-Δh). Po onsecie zaktualizowany
           jest `disability_score` agenta.

        3. **Mortality** — bazowy hazard z GUS modyfikowany przez:
           - disability_score (waga DW jak dotąd),
           - mnożnik Coxa exp(Σ γ_d · H_cum[d]) — TYLKO dla aktywnych chorób,
           - globalny mortality_multiplier (kalibracja).
        """
        alive_citizens = [c for c in self.citizens.values() if c.alive]
        disease_weights = self.disease_model.get_all_disability_weights()
        diseases_list = self.disease_model.diseases
        deaths: List[int] = []

        worker_count = min(self.parallel_workers, max(1, len(alive_citizens)))
        if worker_count > 1 and len(alive_citizens) > worker_count * 2:
            chunk_size = max(1, len(alive_citizens) // worker_count)
            chunks = self._chunked_list(alive_citizens, chunk_size)
            with ThreadPool(worker_count) as pool:
                results = pool.starmap(
                    self._process_death_batch,
                    [(chunk, disease_weights, diseases_list) for chunk in chunks]
                )
            for batch_deaths in results:
                deaths.extend(batch_deaths)
        else:
            deaths = self._process_death_batch(alive_citizens, disease_weights, diseases_list)

        # Usuń zmarłych z gospodarstw
        for citizen_id in deaths:
            citizen = self.citizens[citizen_id]
            citizen.alive = False
            household = self.households.get(citizen.household_id)
            if household:
                household.remove_member(citizen_id)
    
    def handle_births(self) -> None:
        """Process births for eligible females using fertility table."""
        fertile_females = [c for c in self.citizens.values() if c.alive and c.sex == "female"]

        worker_count = min(self.parallel_workers, max(1, len(fertile_females)))
        mothers: List[Citizen] = []

        if worker_count > 1 and len(fertile_females) > worker_count * 2:
            chunk_size = max(1, len(fertile_females) // worker_count)
            chunks = self._chunked_list(fertile_females, chunk_size)
            with ThreadPool(worker_count) as pool:
                results = pool.map(self._process_birth_batch, chunks)
            for batch_mothers in results:
                mothers.extend(batch_mothers)
        else:
            mothers = self._process_birth_batch(fertile_females)

        # Create newborns
        for mother in mothers:
            newborn_sex = mother.rng.choice(["male", "female"])
            
            newborn = Citizen(
                sex=newborn_sex,
                age_months=0,
                household_id=mother.household_id,
                zone_id=mother.zone_id,
                diseases=self.disease_model.get_initial_diseases(),
            )
            
            # Newborns have no risk factors
            newborn.risk_factors = {rf: 0 for rf in Citizen.DEFAULT_RISK_FACTORS}
            
            self.citizens[newborn.id] = newborn
            
            household = self.households.get(mother.household_id)
            if household:
                household.add_member(newborn.id)
    
    def handle_household_splits(self) -> None:
        """
        Handle young adults leaving to form new households.
        
        Adults aged 25+ may leave their current household
        with some probability to form new households.
        """
        potential_movers = []
        
        for citizen in self.citizens.values():
            if (citizen.alive and 
                citizen.age_years >= 25 and 
                self.rng.random() < self.household_split_probability):
                potential_movers.append(citizen)
        
        # Move to new households
        for citizen in potential_movers:
            old_household = self.households.get(citizen.household_id)
            if old_household:
                old_household.remove_member(citizen.id)
            
            # Create new household
            zone_id = old_household.zone_id if old_household else 1
            new_household = Household(zone_id)
            self.households[new_household.id] = new_household
            new_household.add_member(citizen.id)
            citizen.household_id = new_household.id
    
    def _count_active_households(self) -> int:
        """Count households that still have at least one living member."""
        return sum(
            1 for h in self.households.values()
            if h.size() > 0 and any(
                self.citizens[m_id].alive
                for m_id in h.members
                if m_id in self.citizens
            )
        )
    
    def _chunked_list(self, items: List, chunk_size: int) -> List[List]:
        """Split an iterable into chunks for parallel processing."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _age_citizen_batch(self, batch: List[Citizen]) -> None:
        for citizen in batch:
            if citizen.alive:
                citizen.age_one_month()

    def _process_death_batch(
        self,
        batch: List[Citizen],
        disease_weights: Dict[str, float],
        diseases_list: List[str],
    ) -> List[int]:
        dead_ids: List[int] = []
        for citizen in batch:
            if not citizen.alive:
                continue

            new_onset = False
            for disease in diseases_list:
                if disease not in citizen.cumulative_hazard:
                    citizen.cumulative_hazard[disease] = 0.0
                if disease not in citizen.diseases:
                    citizen.diseases[disease] = 0

                delta_h = self.disease_model.monthly_hazard_increment(
                    disease=disease,
                    age_years=citizen.age_years,
                    risk_factors=citizen.risk_factors,
                )
                citizen.cumulative_hazard[disease] += delta_h

                if citizen.diseases[disease] == 0 and delta_h > 0.0:
                    onset_prob = 1.0 - math.exp(-delta_h)
                    if citizen.rng.random() < onset_prob:
                        citizen.diseases[disease] = 1
                        new_onset = True

            if new_onset:
                citizen.compute_disability_score(disease_weights)

            base_mortality = self._get_mortality_rate(citizen.age_years, citizen.sex)
            disease_multiplier = 1.0 + 0.04 * citizen.disability_score
            cox_log = self.disease_model.cox_mortality_log_hazard(
                citizen.diseases, citizen.cumulative_hazard
            )
            cox_multiplier = math.exp(cox_log)

            total_mortality = (
                base_mortality
                * disease_multiplier
                * cox_multiplier
                * self.mortality_multiplier
            )
            total_mortality = min(max(total_mortality, 0.0), 1.0)

            if citizen.rng.random() < total_mortality:
                dead_ids.append(citizen.id)

        return dead_ids

    def _process_birth_batch(self, batch: List[Citizen]) -> List[Citizen]:
        mothers: List[Citizen] = []
        for citizen in batch:
            if citizen.alive and citizen.sex == "female":
                age_years = citizen.age_years
                annual_fertility = self._get_fertility_rate(age_years) * self.fertility_rate
                monthly_fertility = annual_fertility / 12.0
                disease_reduction = 1.0 - (0.02 * citizen.num_conditions() + 0.04 * citizen.disability_score)
                disease_reduction = max(disease_reduction, 0.7)
                monthly_fertility *= disease_reduction

                if citizen.rng.random() < monthly_fertility:
                    mothers.append(citizen)
        return mothers

    def _update_health_batch(self, batch: List[Citizen]) -> None:
        for citizen in batch:
            citizen.update_health_state(self.disease_model)

    def collect_yearly_stats(self, year: int) -> None:
        """
        Collect population statistics for a given year.
        
        Args:
            year: Year number (1-50)
        """
        alive_citizens = [c for c in self.citizens.values() if c.alive]
        
        if not alive_citizens:
            self.yearly_stats[year] = {
                "total_population": 0,
                "num_households": 0,
                "average_household_size": 0,
                "num_males": 0,
                "num_females": 0,
                "age_pyramid": {},
            }
            return
        
        # Basic stats
        males = [c for c in alive_citizens if c.sex == "male"]
        females = [c for c in alive_citizens if c.sex == "female"]
        
        # Active households with at least one living member
        active_households = [
            h for h in self.households.values() 
            if h.size() > 0 and any(
                self.citizens[m_id].alive for m_id in h.members 
                if m_id in self.citizens
            )
        ]
        
        avg_household_size = (
            sum(h.size() for h in active_households) / len(active_households)
            if active_households else 0
        )
        
        # Age pyramid (5-year bins)
        age_pyramid = self._build_age_pyramid(alive_citizens)
        
        self.yearly_stats[year] = {
            "total_population": len(alive_citizens),
            "num_households": len(active_households),
            "num_household_objects": len(self.households),
            "average_household_size": avg_household_size,
            "num_males": len(males),
            "num_females": len(females),
            "age_pyramid": age_pyramid,
            "multimorbidity_count": sum(
                1 for c in alive_citizens if c.has_multimorbidity()
            ),
            "average_disability_score": (
                sum(c.disability_score for c in alive_citizens) / len(alive_citizens)
            ) if alive_citizens else 0.0,
        }
    
    def _build_age_pyramid(self, citizens: List[Citizen]) -> Dict[str, Dict[str, int]]:
        """
        Build age pyramid data for visualization (5-year age bins from 0 to 90+).
        
        Args:
            citizens: List of citizens to include
        
        Returns:
            Dictionary with age bins and male/female counts
        """
        pyramid = {}
        
        # Create 5-year age bins from 0 to 90+
        for start_age in range(0, 90, 5):
            end_age = start_age + 5
            bin_name = f"{start_age}-{end_age-1}"
            
            males = sum(
                1 for c in citizens 
                if c.sex == "male" and start_age <= c.age_years < end_age
            )
            females = sum(
                1 for c in citizens 
                if c.sex == "female" and start_age <= c.age_years < end_age
            )
            
            pyramid[bin_name] = {"male": males, "female": females}
        
        # Add 90-94 age group
        bin_name = "90-94"
        males = sum(
            1 for c in citizens 
            if c.sex == "male" and 90 <= c.age_years < 95
        )
        females = sum(
            1 for c in citizens 
            if c.sex == "female" and 90 <= c.age_years < 95
        )
        pyramid[bin_name] = {"male": males, "female": females}
        
        # Add 95-99 age group
        bin_name = "95-99"
        males = sum(
            1 for c in citizens 
            if c.sex == "male" and 95 <= c.age_years < 100
        )
        females = sum(
            1 for c in citizens 
            if c.sex == "female" and 95 <= c.age_years < 100
        )
        pyramid[bin_name] = {"male": males, "female": females}
        
        # Add 100+ age group
        bin_name = "100+"
        males = sum(
            1 for c in citizens 
            if c.sex == "male" and c.age_years >= 100
        )
        females = sum(
            1 for c in citizens 
            if c.sex == "female" and c.age_years >= 100
        )
        pyramid[bin_name] = {"male": males, "female": females}
        
        return pyramid
    
    def get_statistics_summary(self) -> str:
        """Get a text summary of simulation statistics."""
        lines = ["=" * 70]
        lines.append("SIMULATION STATISTICS")
        lines.append("=" * 70)
        
        if not self.yearly_stats:
            lines.append("No statistics collected yet.")
            return "\n".join(lines)
        
        final_year = max(self.yearly_stats.keys())
        stats = self.yearly_stats[final_year]
        
        lines.append(f"Year: {final_year}")
        lines.append(f"Total Population: {stats['total_population']}")
        total_pop = max(stats['total_population'], 1)
        lines.append(f"Male: {stats['num_males']} ({stats['num_males']/total_pop*100:.1f}%)")
        lines.append(f"Female: {stats['num_females']} ({stats['num_females']/total_pop*100:.1f}%)")
        lines.append(f"Active Households: {stats['num_households']}")
        lines.append(f"Total Household objects: {stats.get('num_household_objects', len(self.households))}")
        lines.append(f"Average Household Size: {stats['average_household_size']:.2f}")
        lines.append(f"Multimorbidity Cases: {stats.get('multimorbidity_count', 0)}")
        lines.append(f"Average Disability Score: {stats.get('average_disability_score', 0.0):.3f}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
