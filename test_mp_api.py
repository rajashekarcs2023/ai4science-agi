"""
Test Materials Project API integration
"""

from src.discovery import NovelMaterialsDiscovery
import pandas as pd

print("\n" + "="*70)
print("🔬 TESTING MATERIALS PROJECT API")
print("="*70 + "\n")

# Test API key
API_KEY = "ZeBYRahOYGbCrmMYtgNcb9Uwm7FUZU2u"

# Initialize discovery engine
print("1️⃣ Initializing Materials Project client...")
discovery = NovelMaterialsDiscovery(api_key=API_KEY)

if discovery.mp_client is None:
    print("❌ FAILED: MP client not initialized")
    exit(1)

print("✅ MP client initialized successfully\n")

# Test with known materials
print("2️⃣ Testing with known battery materials...")
test_formulas = [
    "LiCoO2",      # Known: LiCoO2 cathode
    "LiFePO4",     # Known: LiFePO4 cathode
    "LiMn2O4",     # Known: LiMn2O4 spinel
    "Li2FeSiO4",   # Might exist
    "LiNiCoMnO2",  # Mixed metal oxide
]

print(f"Testing {len(test_formulas)} formulas...\n")

results = discovery.validate_via_mp(test_formulas, max_queries=len(test_formulas))

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70 + "\n")

for idx, row in results.iterrows():
    print(f"Formula: {row['formula']}")
    print(f"  Exists in MP: {row['mp_exists']}")
    
    if row['mp_exists']:
        print(f"  MP ID: {row.get('mp_id', 'N/A')}")
        print(f"  MP Formula: {row.get('mp_formula', 'N/A')}")
        print(f"  Formation Energy: {row.get('mp_formation_energy', 'N/A')} eV/atom")
        print(f"  Energy Above Hull: {row.get('mp_energy_above_hull', 'N/A')} eV/atom")
        print(f"  Stable: {row.get('mp_stable', 'N/A')}")
    else:
        if 'error' in row:
            print(f"  Error: {row['error']}")
        else:
            print(f"  ✨ Novel material - not in MP database!")
    
    print()

print("="*70)
print("✅ TEST COMPLETE")
print("="*70)

# Summary
n_exists = results['mp_exists'].sum()
n_total = len(results)

print(f"\nSummary:")
print(f"  Total tested: {n_total}")
print(f"  Found in MP: {n_exists}")
print(f"  Novel: {n_total - n_exists}")
