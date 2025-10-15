import matplotlib
# Set backend for GUI display
try:
    matplotlib.use('TkAgg')
except ImportError:
    matplotlib.use('Agg')
    print("⚠️  Using Agg backend - plots will be saved but not displayed")
import matplotlib.pyplot as plt

def plot_nested_cartesian_planes(origin, axis_range, depth=5, scale=0.7):
    """
    Plots nested Cartesian coordinate planes centered at the same origin.
    
    Parameters:
    - origin: tuple (x, y) center of the coordinate systems
    - axis_range: range of the outermost coordinate system (from -axis_range to +axis_range)
    - depth: number of nested coordinate planes
    - scale: scaling factor for each nested plane
    """
    cx, cy = origin
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colors for different levels
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(depth):
        current_range = axis_range * (scale ** i)
        color = colors[i % len(colors)]
        alpha = 1.0 - (i * 0.1)  # Fade outer planes
        
        # Draw x-axis for this level
        ax.arrow(cx - current_range, cy, 2 * current_range, 0, 
                head_width=current_range*0.03, head_length=current_range*0.05, 
                fc=color, ec=color, alpha=alpha, linewidth=2)
        
        # Draw y-axis for this level
        ax.arrow(cx, cy - current_range, 0, 2 * current_range, 
                head_width=current_range*0.03, head_length=current_range*0.05, 
                fc=color, ec=color, alpha=alpha, linewidth=2)
        
        # Add tick marks on axes
        tick_spacing = current_range / 4
        for tick in [-current_range, -current_range/2, current_range/2, current_range]:
            if tick != 0:  # Don't draw tick at origin
                # X-axis ticks
                ax.plot([cx + tick, cx + tick], [cy - current_range*0.02, cy + current_range*0.02], 
                       color=color, alpha=alpha, linewidth=1.5)
                # Y-axis ticks  
                ax.plot([cx - current_range*0.02, cx + current_range*0.02], [cy + tick, cy + tick], 
                       color=color, alpha=alpha, linewidth=1.5)
        
        # Add boundary stroke for this coordinate plane
        boundary_width = 2 * current_range
        boundary_height = 2 * current_range
        lower_left = (cx - current_range, cy - current_range)
        boundary_rect = plt.Rectangle(lower_left, boundary_width, boundary_height, 
                                    fill=False, edgecolor=color, alpha=alpha, 
                                    linewidth=2.5, linestyle='-')
        ax.add_patch(boundary_rect)
        
        # Add labels for this coordinate system level
        ax.text(cx + current_range*1.1, cy, f'X{i+1}', fontsize=12-i, 
               color=color, alpha=alpha, weight='bold')
        ax.text(cx, cy + current_range*1.1, f'Y{i+1}', fontsize=12-i, 
               color=color, alpha=alpha, weight='bold')
        
        # Add coordinate plane label in corner
        ax.text(cx - current_range*0.95, cy + current_range*0.9, f'Plane {i+1}', 
               fontsize=10-i, color=color, alpha=alpha, weight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
        
        # Add grid for this level (optional)
        if i < 3:  # Only show grid for first 3 levels to avoid clutter
            grid_spacing = current_range / 2
            for grid_pos in [-current_range, -grid_spacing, grid_spacing, current_range]:
                if grid_pos != 0:
                    ax.axhline(y=cy + grid_pos, color=color, alpha=alpha*0.3, linestyle='--', linewidth=0.5)
                    ax.axvline(x=cx + grid_pos, color=color, alpha=alpha*0.3, linestyle='--', linewidth=0.5)

    # Mark the origin point
    ax.plot(cx, cy, 'ko', markersize=8, label='Common Origin')
    
    ax.set_aspect('equal')
    ax.set_xlim(cx - axis_range*1.2, cx + axis_range*1.2)
    ax.set_ylim(cy - axis_range*1.2, cy + axis_range*1.2)
    ax.set_title("Nested Cartesian Coordinate Planes", fontsize=16, weight='bold')
    ax.legend(loc='upper right')
    
    # Remove the default axes to avoid confusion with our nested ones
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_nested_cartesian_planes(origin=(0, 0), axis_range=4, depth=4, scale=0.5)
