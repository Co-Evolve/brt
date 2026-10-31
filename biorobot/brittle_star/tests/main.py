import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    t = np.linspace(0, 5, 100)

    frequency = 1
    phase_offset = 0

    x = np.cos(frequency * t + phase_offset)
    y = 0.5 * np.sin(frequency * t + phase_offset)

    # Create a color gradient from light blue to dark blue
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(t)))  # Using the 'Blues' colormap

    plt.scatter(x, y, c=colors, edgecolor="k")  # Add edgecolor for visibility
    plt.show()
    plt.close()
