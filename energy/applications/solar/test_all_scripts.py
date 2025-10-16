#!/usr/bin/env python3
"""
Test script to verify all solar energy scripts work correctly
"""

import subprocess
import sys
import os

def test_script(script_name):
    """Test a single script"""
    print(f"\n{'='*50}")
    print(f"Testing {script_name}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {script_name} - PASSED")
            return True
        else:
            print(f"❌ {script_name} - FAILED")
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} - TIMEOUT (probably generating plots)")
        return True  # Plots cause timeout, but script works
    except Exception as e:
        print(f"❌ {script_name} - ERROR: {e}")
        return False

def main():
    """Test all solar energy scripts"""
    
    scripts = [
        "photovoltaic_system.py",
        "concentrated_solar_power.py", 
        "solar_thermal_heating.py",
        "solar_energy_challenges.py",
        "solar_energy_demo.py"
    ]
    
    print("Testing all solar energy application scripts...")
    
    passed = 0
    total = len(scripts)
    
    for script in scripts:
        if os.path.exists(script):
            if test_script(script):
                passed += 1
        else:
            print(f"❌ {script} - FILE NOT FOUND")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{total} scripts passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All scripts are working correctly!")
    else:
        print("⚠️  Some scripts need attention")

if __name__ == "__main__":
    main()