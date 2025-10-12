"""
Material explanation system - chemistry knowledge without LLM
"""

def explain_material_chemistry(formula: str) -> str:
    """
    Generate chemistry explanation based on formula
    Uses domain knowledge, not LLMs (more reliable)
    """
    
    explanation = []
    
    # Detect element types
    if 'Li' in formula:
        explanation.append("**Lithium (Li):** Mobile ion in battery - shuttles between electrodes during charge/discharge")
    
    if 'Ni' in formula:
        explanation.append("**Nickel (Ni):** High capacity transition metal - used in high-energy cathodes like NMC")
    
    if 'Co' in formula:
        explanation.append("**Cobalt (Co):** Improves structural stability - used in LiCoO₂ (first commercial Li-ion cathode)")
    
    if 'Mn' in formula:
        explanation.append("**Manganese (Mn):** Low-cost, stable - safer than Co, used in LiMn₂O₄ spinel")
    
    if 'Fe' in formula:
        explanation.append("**Iron (Fe):** Abundant and cheap - LiFePO₄ is low-cost, safe cathode")
    
    if 'P' in formula and 'O' in formula:
        explanation.append("**Phosphate (PO₄):** Strong P-O bonds provide thermal stability and safety")
    
    if 'F' in formula:
        explanation.append("**Fluorine (F):** Highly electronegative - increases voltage by strengthening ionic bonds")
    
    if 'S' in formula and 'O' in formula:
        explanation.append("**Sulfate (SO₄):** Alternative to phosphate - can provide high voltage in fluorosulfates")
    
    if 'Ti' in formula:
        explanation.append("**Titanium (Ti):** Structural dopant - improves cycle life and safety")
    
    if 'Al' in formula:
        explanation.append("**Aluminum (Al):** Common dopant - reduces cation mixing, improves stability")
    
    if 'V' in formula:
        explanation.append("**Vanadium (V):** Multi-valent - can provide high voltage but may have safety concerns")
    
    if 'Na' in formula:
        explanation.append("**Sodium (Na):** Na-ion alternative to Li-ion - more abundant, lower cost")
    
    # Detect structure types
    if 'PO4F' in formula or ('P' in formula and 'F' in formula):
        explanation.append("\n**🔬 Structure Type:** Tavorite fluorophosphate - cutting-edge high-voltage cathode")
    elif 'SO4F' in formula:
        explanation.append("\n**🔬 Structure Type:** Fluorosulfate - emerging high-voltage cathode material")
    elif 'Mn2O4' in formula:
        explanation.append("\n**🔬 Structure Type:** Spinel - 3D Li diffusion pathways, good rate capability")
    elif 'PO4' in formula:
        explanation.append("\n**🔬 Structure Type:** Olivine/Phosphate - thermally stable, safe")
    
    return "\n\n".join(explanation) if explanation else "Standard battery cathode material."


def get_material_category(formula: str, voltage: float) -> dict:
    """
    Categorize material and provide context
    """
    category = {
        'type': 'Unknown',
        'voltage_class': 'Standard',
        'commercial_analog': None,
        'pros': [],
        'cons': []
    }
    
    # Voltage classification
    if voltage >= 4.5:
        category['voltage_class'] = 'High Voltage (>4.5V)'
        category['pros'].append('High energy density potential')
        category['cons'].append('Requires stable electrolytes')
    elif voltage >= 3.5:
        category['voltage_class'] = 'Medium-High (3.5-4.5V)'
        category['pros'].append('Good balance of voltage and stability')
    else:
        category['voltage_class'] = 'Low-Medium (<3.5V)'
        category['cons'].append('Lower energy density')
    
    # Chemistry classification
    if 'Ni' in formula and 'Co' in formula and 'Mn' in formula:
        category['type'] = 'NMC-type (Nickel-Manganese-Cobalt)'
        category['commercial_analog'] = 'LiNi₀.₃₃Mn₀.₃₃Co₀.₃₃O₂ (NMC-111)'
        category['pros'].append('High capacity, used in EVs')
        category['cons'].append('Contains expensive cobalt')
    
    elif 'Ni' in formula and 'Mn' in formula:
        category['type'] = 'High-Nickel Cathode'
        category['commercial_analog'] = 'LiNi₀.₈Mn₀.₁Co₀.₁O₂ (NMC-811)'
        category['pros'].append('High energy density')
        category['cons'].append('Can be structurally unstable')
    
    elif 'Fe' in formula and 'P' in formula:
        category['type'] = 'Iron Phosphate'
        category['commercial_analog'] = 'LiFePO₄ (Olivine)'
        category['pros'].extend(['Low cost', 'Very safe', 'Long cycle life'])
        category['cons'].append('Lower voltage (~3.45V)')
    
    elif 'Mn' in formula and 'O4' in formula:
        category['type'] = 'Manganese Spinel'
        category['commercial_analog'] = 'LiMn₂O₄'
        category['pros'].extend(['Low cost', 'Good rate capability'])
        category['cons'].append('Mn dissolution issue')
    
    elif 'Co' in formula and not 'Ni' in formula and not 'Mn' in formula:
        category['type'] = 'Cobalt Oxide'
        category['commercial_analog'] = 'LiCoO₂'
        category['pros'].extend(['First commercial cathode', 'Good performance'])
        category['cons'].extend(['Expensive', 'Safety concerns at high voltage'])
    
    elif 'F' in formula and 'P' in formula:
        category['type'] = 'Fluorophosphate (Next-Gen)'
        category['commercial_analog'] = 'Li₂FePO₄F (Research Stage)'
        category['pros'].extend(['Very high voltage potential', 'Thermally stable'])
        category['cons'].append('Not yet commercialized')
    
    elif 'Na' in formula:
        category['type'] = 'Sodium-ion Cathode'
        category['commercial_analog'] = 'Na-ion batteries (Emerging)'
        category['pros'].extend(['Abundant', 'Lower cost than Li'])
        category['cons'].append('Lower energy density than Li-ion')
    
    return category


if __name__ == "__main__":
    # Test
    test_formula = "Li2NiPO4F"
    print(explain_material_chemistry(test_formula))
    print("\n" + "="*50)
    category = get_material_category(test_formula, 4.8)
    print(f"Type: {category['type']}")
    print(f"Voltage Class: {category['voltage_class']}")
    print(f"Pros: {category['pros']}")
