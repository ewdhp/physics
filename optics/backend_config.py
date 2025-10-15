#!/usr/bin/env python3
"""
Matplotlib Backend Configuration Utility
========================================

This utility helps configure and test matplotlib backends for the physics simulations.
It automatically detects available GUI backends and configures the best option.
"""

import matplotlib
import sys
import os


def detect_available_backends():
    """Detect which GUI backends are available on the system."""
    available_gui_backends = []
    all_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WebAgg']
    
    for backend in all_backends:
        try:
            matplotlib.use(backend)
            # Test if backend actually works
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.close(fig)
            available_gui_backends.append(backend)
        except (ImportError, RuntimeError):
            continue
    
    return available_gui_backends


def configure_backend(backend=None):
    """Configure matplotlib to use the specified or best available backend."""
    
    if backend:
        try:
            matplotlib.use(backend)
            print(f"✅ Successfully set backend to: {backend}")
            return backend
        except (ImportError, RuntimeError) as e:
            print(f"❌ Failed to set backend {backend}: {e}")
            return None
    
    # Auto-detect best backend
    available = detect_available_backends()
    
    if not available:
        print("⚠️  No GUI backends available, using Agg (non-interactive)")
        matplotlib.use('Agg')
        return 'Agg'
    
    # Preference order: TkAgg (most stable) -> Qt5Agg -> others
    preferred_order = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WebAgg']
    
    for preferred in preferred_order:
        if preferred in available:
            matplotlib.use(preferred)
            print(f"✅ Auto-configured backend: {preferred}")
            return preferred
    
    # Fallback to first available
    backend = available[0]
    matplotlib.use(backend)
    print(f"✅ Using backend: {backend}")
    return backend


def test_backend():
    """Test the current backend with a simple plot."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print(f"🧪 Testing backend: {matplotlib.get_backend()}")
        
        # Create test plot
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Backend Test: {matplotlib.get_backend()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if matplotlib.get_backend() == 'Agg':
            # Save plot for non-GUI backend
            plt.savefig('backend_test.png', dpi=150, bbox_inches='tight')
            print("📄 Plot saved as 'backend_test.png' (non-GUI backend)")
            plt.close()
        else:
            print("🖥️  Plot window should appear. Close it to continue...")
            plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False


def main():
    """Main configuration interface."""
    print("🎨 Matplotlib Backend Configuration")
    print("=" * 40)
    
    print(f"Current backend: {matplotlib.get_backend()}")
    
    if len(sys.argv) > 1:
        requested_backend = sys.argv[1]
        print(f"Attempting to set backend to: {requested_backend}")
        backend = configure_backend(requested_backend)
    else:
        print("Auto-detecting best backend...")
        backend = configure_backend()
    
    if backend:
        print(f"\n📋 Backend Configuration Summary:")
        print(f"   Backend: {backend}")
        print(f"   Interactive: {'Yes' if backend != 'Agg' else 'No'}")
        
        # Test the backend
        print(f"\n🧪 Testing backend...")
        if test_backend():
            print(f"✅ Backend {backend} is working correctly!")
        else:
            print(f"❌ Backend {backend} test failed")
    
    print(f"\n📚 Available backends on this system:")
    available = detect_available_backends()
    if available:
        for b in available:
            print(f"   • {b}")
    else:
        print("   • Agg (non-interactive only)")
    
    print(f"\n💡 Usage:")
    print(f"   python backend_config.py           # Auto-detect best backend")
    print(f"   python backend_config.py TkAgg     # Force specific backend")
    print(f"   python backend_config.py Qt5Agg    # Try Qt5 backend")


if __name__ == "__main__":
    main()