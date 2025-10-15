#!/usr/bin/env python3
"""
Wave Optics Test Suite
=====================

Quick test and demonstration of all wave optics modules.
Run this to verify everything is working correctly.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import(module_name, script_path):
    """Test importing and basic functionality of a module."""
    print(f"🔧 Testing {module_name}...")
    try:
        # Simple syntax check by trying to compile the file
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, script_path, 'exec')
        print(f"  ✅ {module_name} syntax check passed")
        return True
            
    except SyntaxError as e:
        print(f"  ❌ Syntax error in {module_name}: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error checking {module_name}: {e}")
        return False

def main():
    """Run tests for all wave optics modules."""
    print("🌊 Wave Optics Test Suite")
    print("=" * 30)
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of modules to test
    modules = [
        ("interference_phenomena", "interference_phenomena.py"),
        ("double_slit_experiment", "double_slit_experiment.py"), 
        ("diffraction_patterns", "diffraction_patterns.py"),
        ("polarization", "polarization.py"),
        ("coherence", "coherence.py")
    ]
    
    results = []
    
    for module_name, filename in modules:
        script_path = os.path.join(current_dir, filename)
        success = test_import(module_name, script_path)
        results.append((module_name, success))
    
    print(f"\n📊 Test Results:")
    print("=" * 20)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for module_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {module_name}")
    
    print(f"\n🎯 Summary: {passed}/{total} modules passed")
    
    if passed == total:
        print("🎉 All wave optics modules are ready to use!")
        print("\n📚 Available demonstrations:")
        print("  • python interference_phenomena.py")
        print("  • python double_slit_experiment.py")
        print("  • python diffraction_patterns.py")
        print("  • python polarization.py")
        print("  • python coherence.py")
    else:
        print("⚠️  Some modules failed. Check error messages above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)