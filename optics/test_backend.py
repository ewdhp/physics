#!/usr/bin/env python3
"""Test matplotlib backend setup"""

import matplotlib
print(f"Current backend: {matplotlib.get_backend()}")

# Set TkAgg backend
try:
    matplotlib.use('TkAgg')
    print(f"Backend set to: {matplotlib.get_backend()}")
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a simple test plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Matplotlib Backend Test - TkAgg')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("✅ Matplotlib with TkAgg is working!")
    print("✅ Plot window should appear")
    print("Close the plot window to continue...")
    
    plt.show()
    
except Exception as e:
    print(f"❌ Error with TkAgg backend: {e}")
    print("Falling back to Agg backend for non-GUI use")
    matplotlib.use('Agg')
    print(f"Backend now: {matplotlib.get_backend()}")