from core.risk_manager import RiskManager

class RealEstateAgent(BaseAgent):
    """
    Evaluates real estate opportunities and manages property portfolio.
    
    Responsibilities:
    1. Cash flow analysis (Cap Rate, CoC Return).
    2. Regional appreciation forecasting.
    3. Liquidity stress testing via Central RiskManager.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.risk_manager = RiskManager(config)
        self.savings = 0
        self.debt = 0
    
    def update_market_state(self, market_data: Dict[str, Any]):
        """Updates regional price indices and holistic financial health."""
        self.savings = market_data.get('savings', self.savings)
        self.debt = market_data.get('debt', self.debt)

    def evaluate_property(self, property_details: Dict) -> Dict:
        """
        Analyzes a potential property acquisition with neighborhood, risk, and COL governance.
        """
        price = property_details.get('price', 0)
        rent = property_details.get('monthly_rent', 0)
        expenses = property_details.get('monthly_expenses', 0)
        zip_code = property_details.get('zip_code', "00000")
        state = property_details.get('state', "US")
        
        # 1. Financial Analysis
        noi = (rent - expenses) * 12
        cap_rate = noi / price if price > 0 else 0
        
        # 2. Neighborhood Analysis (Census-Powered)
        nb_stats = self.analyze_neighborhood(zip_code)
        nb_score = nb_stats.get('neighborhood_score', 5)
        
        # 3. Risk Governance (DTI, Liquidity, & COL Normalization)
        risk_report = self.check_acquisition_risk(price, property_details.get('down_payment', 0), state)
        is_safe = risk_report.get('is_safe', False)
        
        # 4. Final Recommendation
        is_approved = cap_rate > 0.08 and nb_score >= 7 and is_safe
        
        return {
            "noi": noi,
            "cap_rate": cap_rate,
            "neighborhood_score": nb_score,
            "risk_report": risk_report,
            "recommendation": "BUY" if is_approved else "VETO (RISK)" if not is_safe else "PASS"
        }

    def check_acquisition_risk(self, property_price: float, down_payment: float, state: str = "US") -> Dict:
        """
        Holistic Risk Check: DTI, Liquidity Reserve, and COL Normalization.
        """
        projected_loan = property_price - down_payment
        monthly_mortgage = (projected_loan * 0.07) / 12 # Mock 7% interest
        
        # 1. COL Adjustment: Normalize income based on state purchasing power
        col_index = self._get_state_col_index(state)
        # Higher COL = lower purchasing power of the same dollar
        base_income = 8000 # Mock monthly income
        normalized_income = base_income / (col_index / 100.0)
        
        # 2. Calculate DTI (Debt-to-Income)
        total_monthly_debt = self.debt + monthly_mortgage
        dti = total_monthly_debt / normalized_income
        
        # 3. Liquidity Check
        remaining_savings = self.savings - down_payment
        has_reserve = remaining_savings > (monthly_mortgage * 6)
        
        is_safe = dti < 0.43 and has_reserve
        
        return {
            "is_safe": is_safe,
            "dti": dti,
            "normalized_income": normalized_income,
            "col_index": col_index,
            "reason": f"DTI {dti:.2f} safe at {state} COL index {col_index}" if is_safe else "DTI too high for regional COL"
        }

    def _get_state_col_index(self, state: str) -> float:
        """
        Mock for State Cost of Living Index (100 = National Average).
        In production, this would call MERIC or C2ER database.
        """
        col_indices = {
            "CA": 138.5,
            "NY": 126.7,
            "TX": 93.0,
            "OH": 91.3,
            "FL": 102.8
        }
        return col_indices.get(state, 100.0)

    def analyze_neighborhood(self, zip_code: str) -> Dict:
        """
        Calculates a 'Neighborhood Score' using Census demographic and economic data.
        """
        census_data = self._fetch_census_metrics(zip_code)
        
        # Weighted Scoring Logic (Placeholder)
        score = 0
        if census_data.get('pop_growth', 0) > 0.02: score += 3 # Boom town
        if census_data.get('median_income', 0) > 60000: score += 3 # Stable renter base
        if census_data.get('owner_occupancy', 0) > 0.50: score += 2 # Pride of ownership
        if census_data.get('edu_level', 0) > 0.30: score += 2 # High-skill workforce
        
        return {
            "zip_code": zip_code,
            "neighborhood_score": min(score, 10),
            "metrics": census_data
        }

    def _fetch_census_metrics(self, zip_code: str) -> Dict:
        """
        Mock fetch for U.S. Census Bureau ACS 5-Year Data.
        In production, this would call: api.census.gov/data/2022/acs/acs5
        """
        # Placeholder data - real implementation would use 'requests' and 'pandas'
        return {
            "pop_growth": 0.025,
            "median_income": 72000,
            "owner_occupancy": 0.62,
            "edu_level": 0.45
        }

    def optimize_rent(self, current_rent: float, market_vacancy_rate: float) -> Dict:
        """
        Calculates Lease Elasticity: Is a rent hike worth the vacancy risk?
        """
        proposed_rent = current_rent * 1.05 # Standard 5% increase
        market_rent = self._get_local_rent_comps()
        
        # Risk Math: If vacancy is high (>7%), rent hikes are risky
        risk_factor = market_vacancy_rate * 10
        recommended_rent = proposed_rent if risk_factor < 0.5 else current_rent
        
        return {
            "current_rent": current_rent,
            "market_rent": market_rent,
            "recommended_rent": recommended_rent,
            "hike_recommendation": proposed_rent > current_rent
        }

    def predict_maintenance(self, property_age: int, last_major_reno: int) -> Dict:
        """
        Predicts sinking fund requirements for major components.
        """
        # Logic: AC (15yrs), Roof (25yrs), Water Heater (10yrs)
        components = {
            "HVAC": 15 - (property_age % 15),
            "Roof": 25 - (property_age % 25),
            "Plumbing": 40 - (property_age % 40)
        }
        
        # Calculate monthly savings required to cover the next failure
        next_failure = min(components.values())
        suggested_reserve = 500 if next_failure < 2 else 150 # Aggressive if failure is near
        
        return {
            "years_until_major_capx": next_failure,
            "monthly_reserve_target": suggested_reserve,
            "component_health": components
        }

    def _get_local_rent_comps(self) -> float:
        # Mock for Rentometer/Zillow Rent API
        return 2100.0

    def generate_proposals(self) -> List[Dict]:
        """
        Scans integrated listings and generates buy OR optimization proposals.
        """
        # Example of an optimization proposal for an existing property
        return [{
            "type": "RENT_OPTIMIZATION",
            "property_id": "PROP_001",
            "action": "RAISE_RENT",
            "amount": 150.0,
            "reason": "Market gap detected via Comps Analysis"
        }]

    def execute_instruction(self, instruction: Dict[str, Any]):
        # Mock execution: "Contact Agent" or "Submit Offer"
        pass
