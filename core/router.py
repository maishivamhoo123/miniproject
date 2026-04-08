class AlloyRouter:
    @staticmethod
    def identify_system(input_dict: dict) -> str:
        """
        Determines the dominant alloy system based on composition percentages.
        Returns the specific model name corresponding to the highest weight % elements.
        """
        # List of all elemental features in your dataset
        elements = ['Ag','Al','Au','Cd','Co','Cu','Fe','Hf','Mn','Nb',
                    'Ni','Pd','Pt','Ru','Si','Ta','Ti','Zn','Zr']
        
        # Extract only the composition percentages from the input
        composition = {elem: input_dict.get(elem, 0.0) for elem in elements}
        
        # Sort the elements by their percentage in descending order
        sorted_elements = sorted(composition.items(), key=lambda item: item[1], reverse=True)
        
        # Get the top two most abundant elements
        top_two = {sorted_elements[0][0], sorted_elements[1][0]}
        
        # Route to the correct model family based on the top two elements
        if {"Ni", "Ti"}.issubset(top_two):
            return "NiTi_Family"
        elif {"Ni", "Al"}.issubset(top_two):
            return "NiAl_Family"
        elif {"Ni", "Co"}.issubset(top_two):
            return "NiCo_Family"
        else:
            # Fallback model or you could raise a ValueError if the system is unsupported
            # raise ValueError("Unsupported alloy composition for current model registry.")
            return "NiTi_Family"