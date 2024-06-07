import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def animate_performance(losses, interval=100):
    fig, ax = plt.subplots()
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses) * 1.1)
    line, = ax.plot([], [], lw=2, color='#0f68a9')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x = range(frame + 1)
        y = losses[:frame + 1]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, update, frames=len(losses), init_func=init, blit=True, interval=interval)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Training Performance')
    plt.grid(True)

    # Save the animation
    ani.save('output/training_performance.gif', writer='pillow')

    plt.show()

