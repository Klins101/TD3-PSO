import matplotlib.pyplot as plt

def setup_plot_style():
    """latex style"""
    plt.rcParams.update({
        'text.usetex': False,  # Set to True if LaTeX is available
        'font.family': 'serif',
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 18,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def save_plot(fig, filepath, formats=['pdf', 'eps']):
    for fmt in formats:
        save_path = f"{filepath}.{fmt}"
        fig.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"  Saved: {save_path}")


def get_algorithm_color(algorithm):
    colors = {
        'td3': '#ff7f0e',      # Orange
        'td3_pso': '#2ca02c',  # Green
    }
    return colors.get(algorithm.lower(), '#1f77b4')


def get_algorithm_label(algorithm):
    labels = {
        'td3': 'TD3',
        'td3_pso': 'TD3-PSO',
    }
    return labels.get(algorithm.lower(), algorithm.upper())
