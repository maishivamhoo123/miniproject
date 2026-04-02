from typing import Dict

class AlloyRouter:
    @staticmethod
    def identify_system(composition: Dict[str, float]) -> str:
        """
        Dynamically routes to the correct model based on atomic composition.
        """
        ni_ti_sum = composition.get('Ni', 0.0) + composition.get('Ti', 0.0)
        cu_al_sum = composition.get('Cu', 0.0) + composition.get('Al', 0.0)
        
        if ni_ti_sum >= 50.0:
            return "NiTi_Family"
        elif cu_al_sum >= 50.0:
            return "CuAl_Family"
        else:
            return "Generic_Global" # Fallback model