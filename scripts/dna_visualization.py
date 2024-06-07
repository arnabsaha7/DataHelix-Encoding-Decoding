import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# Ensure the correct backend is set for interactive plotting
plt.switch_backend('tkagg')

# Function to get the complementary base
def complement(base):
    return 'T' if base == 'A' else 'A'

# Function to generate 3D coordinates for a double helix
def generate_dna_helix(dna_sequence, pitch=2, radius=1, turns_per_unit=2):
    n = len(dna_sequence)
    z = np.linspace(0, pitch * n / turns_per_unit, n)
    theta = np.linspace(0, 2 * np.pi * n / turns_per_unit, n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    x_complement = radius * np.cos(theta + np.pi)
    y_complement = radius * np.sin(theta + np.pi)
    
    return x, y, z, x_complement, y_complement

# Function to hash a DNA sequence for shorter file names
def hash_dna_sequence(dna_sequence):
    return hashlib.md5(dna_sequence.encode()).hexdigest()

# Function to plot the 3D structure
def plot_dna_helix(dna_sequence, output_dir='output'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z, x_complement, y_complement = generate_dna_helix(dna_sequence)
    
    ax.plot(x, y, z, label='Strand 1', color='#0f68a9')
    ax.plot(x_complement, y_complement, z, label='Strand 2', color='#34073d')
    
    for i in range(len(dna_sequence)):
        ax.plot([x[i], x_complement[i]], [y[i], y_complement[i]], [z[i], z[i]], color='#f7ba2c')
        ax.text(x[i], y[i], z[i], dna_sequence[i], color='black')
        ax.text(x_complement[i], y_complement[i], z[i], complement(dna_sequence[i]), color='black')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D DNA Helix Visualization')
    plt.legend()

    # Save plot with hashed filename
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f'dna_helix_{hash_dna_sequence(dna_sequence)}.png'
    plt.savefig(os.path.join(output_dir, filename))
    plt.show()

# Function to create an animated plot of the DNA helix and save it as a GIF
def animate_dna(dna_sequence, output_dir='output'):
    x, y, z, x_complement, y_complement = generate_dna_helix(dna_sequence)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    line1, = ax.plot([], [], [], label='Strand 1', color='#0974f1')
    line2, = ax.plot([], [], [], label='Strand 2', color='#d397fa')
    lines = [line1, line2]
    text_annotations = []
    
    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, max(z))
        ax.set_title('Animating DNA Helix')
        return lines
    
    def update(frame):
        line1.set_data(x[:frame], y[:frame])
        line1.set_3d_properties(z[:frame])
        line2.set_data(x_complement[:frame], y_complement[:frame])
        line2.set_3d_properties(z[:frame])
        
        while text_annotations:
            txt = text_annotations.pop()
            txt.remove()
        
        for i in range(frame):
            txt1 = ax.text(x[i], y[i], z[i], dna_sequence[i], color='black')
            txt2 = ax.text(x_complement[i], y_complement[i], z[i], complement(dna_sequence[i]), color='black')
            text_annotations.append(txt1)
            text_annotations.append(txt2)
        
        return lines + text_annotations

    frames = len(dna_sequence)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)

    # Show the animation in real-time
    plt.legend()
    plt.show()

    # Save the animation with hashed filename
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f'dna_animation_{hash_dna_sequence(dna_sequence)}.gif'
    ani.save(os.path.join(output_dir, filename), writer=PillowWriter(fps=30))
