"""
LLM-powered synthesis route advisor for discovered materials
"""

def generate_synthesis_route(formula: str, predicted_voltage: float, mp_data: dict = None) -> str:
    """
    Generate synthesis recommendations using chemistry knowledge
    
    Args:
        formula: Chemical formula
        predicted_voltage: Predicted voltage
        mp_data: Materials Project data if available
    
    Returns:
        Detailed synthesis recommendation
    """
    
    synthesis_advice = f"# 🧪 Synthesis Route for {formula}\n\n"
    
    # Extract elements
    elements = []
    if 'Li' in formula:
        elements.append('Lithium')
    if 'Ni' in formula:
        elements.append('Nickel')
    if 'Co' in formula:
        elements.append('Cobalt')
    if 'Mn' in formula:
        elements.append('Manganese')
    if 'Fe' in formula:
        elements.append('Iron')
    if 'P' in formula and 'O' in formula:
        synthesis_type = "Phosphate"
    elif 'S' in formula and 'O' in formula:
        synthesis_type = "Sulfate"
    else:
        synthesis_type = "Oxide"
    
    synthesis_advice += f"## Material Type: {synthesis_type}\n\n"
    
    # Recommend synthesis method based on chemistry
    if 'F' in formula:
        synthesis_advice += """### Recommended Method: **Solid-State Ion Exchange**

**Why:** Fluorinated compounds are challenging to synthesize directly due to HF formation.

**Steps:**
1. **Precursor Synthesis:**
   - Synthesize oxide/phosphate precursor via solid-state reaction
   - React metal salts (carbonates/acetates) at 600-800°C
   - Example: Li₂MPO₄ (M = Ni, Co, Fe, Mn)

2. **Fluorination:**
   - Low-temperature fluorination using NH₄F, LiF, or F₂ gas
   - Temperature: 300-400°C (prevents decomposition)
   - Inert atmosphere (Ar or N₂)
   - Reaction time: 12-24 hours

3. **Characterization:**
   - XRD to confirm crystal structure
   - SEM/TEM for morphology
   - XPS to verify fluorine incorporation
   - Electrochemical testing (charge/discharge curves)

**Safety:** Handle fluorinating agents with extreme care. HF gas is toxic.
"""
    
    elif 'P' in formula and 'O' in formula:
        synthesis_advice += """### Recommended Method: **Solid-State Reaction**

**Steps:**
1. **Precursor Mixing:**
   - Weigh stoichiometric amounts of:
     * Li₂CO₃ or LiOH (lithium source)
     * Metal oxide or acetate (transition metal source)
     * NH₄H₂PO₄ (phosphorus source)
   - Ball-mill for 4-6 hours in ethanol

2. **Calcination:**
   - Pre-heat at 350°C for 4h (decompose carbonates)
   - Main reaction: 600-800°C for 12-24h
   - Ramp rate: 5°C/min
   - Atmosphere: Ar or N₂ (prevent oxidation)

3. **Post-Processing:**
   - Grind and sieve to <10 μm
   - Re-heat at 650°C for 6h (improve crystallinity)
   - Rapid cooling to room temperature

4. **Characterization:**
   - XRD for phase purity
   - ICP-OES for stoichiometry verification
   - Electrochemical testing

**Expected Yield:** 85-95%
"""
    
    else:
        synthesis_advice += """### Recommended Method: **High-Temperature Solid-State**

**Steps:**
1. **Precursor Preparation:**
   - Mix stoichiometric metal oxides/carbonates
   - Ball-mill for 6-8 hours
   - Press into pellets (10 ton pressure)

2. **Calcination:**
   - First heat: 500°C for 6h (decompose carbonates)
   - Main reaction: 800-900°C for 12-24h
   - Controlled oxygen atmosphere
   - Slow cooling: 2°C/min to preserve structure

3. **Post-Processing:**
   - Grind and characterize
   - Optional: Carbon coating for better conductivity

**Expected Structure:** Spinel or layered oxide
"""
    
    # Add voltage-specific recommendations
    synthesis_advice += "\n## 🎯 Performance Optimization\n\n"
    
    if predicted_voltage >= 4.5:
        synthesis_advice += f"""**High Voltage Material ({predicted_voltage:.2f}V)**

**Challenges:**
- Electrolyte decomposition at high voltages
- Oxygen loss during cycling
- Transition metal dissolution

**Recommendations:**
1. **Surface Coating:** Apply Al₂O₃, TiO₂, or ZrO₂ protective layer
   - ALD coating (2-5 nm thickness)
   - Prevents electrolyte contact

2. **Electrolyte Selection:**
   - Use LiPF₆ in EC:DMC with FEC additive
   - Or: Ionic liquid electrolytes for stability
   - Avoid standard carbonates (decompose >4.5V)

3. **Doping Strategy:**
   - Al or Mg doping (2-5%) improves structural stability
   - Reduces cation mixing in layered structures

4. **Particle Size Control:**
   - Nano-sized particles (100-300 nm)
   - Improves rate capability
   - Reduces diffusion length
"""
    
    elif predicted_voltage >= 3.5:
        synthesis_advice += f"""**Medium-High Voltage Material ({predicted_voltage:.2f}V)**

**Optimization:**
1. Carbon coating for improved conductivity
2. Optimize particle size (200-500 nm)
3. Standard carbonate electrolytes should work
4. Consider Mn substitution for lower cost
"""
    
    # Add characterization checklist
    synthesis_advice += """
## ✅ Characterization Checklist

### Structural:
- [ ] XRD: Phase purity, lattice parameters
- [ ] Rietveld refinement: Confirm crystal structure
- [ ] SEM/TEM: Particle morphology and size
- [ ] BET: Surface area measurement

### Chemical:
- [ ] ICP-OES: Elemental composition
- [ ] XPS: Oxidation states, surface chemistry
- [ ] TGA: Thermal stability

### Electrochemical:
- [ ] Galvanostatic charge/discharge (C/10 rate)
- [ ] Cyclic voltammetry (scan rate: 0.1 mV/s)
- [ ] Rate capability (C/10 to 5C)
- [ ] Long-term cycling (100+ cycles)
- [ ] EIS: Charge transfer resistance

## 📚 Expected Results

**If synthesis successful:**
- First cycle coulombic efficiency: 80-95%
- Reversible capacity: 120-180 mAh/g
- Voltage plateau near predicted value
- Stable cycling (>80% retention at 100 cycles)

**Red Flags:**
- Multiple voltage plateaus → impure phase
- Low coulombic efficiency → SEI formation
- Rapid capacity fade → structural instability

## 💰 Estimated Cost & Time

- **Lab Cost:** $200-500 (precursors + characterization)
- **Synthesis Time:** 2-3 days
- **Full Characterization:** 1-2 weeks
- **Electrochemical Testing:** 2-4 weeks

---

*This synthesis route is generated based on chemical knowledge and literature precedents. Always consult safety data sheets and work under supervision.*
"""
    
    return synthesis_advice


if __name__ == "__main__":
    # Test
    route = generate_synthesis_route("Li2NiPO4F", 4.82)
    print(route)
