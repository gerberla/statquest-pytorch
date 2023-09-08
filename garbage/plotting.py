# Plotting
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import collections
import seaborn as sns

import pandas as pd
import time



def live_plot(data_dict, figsize=(7,5), title=''):
    plt.figure(figsize=figsize)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.plot()

np.random.seed(0)

data = np.random.rand(120, 50)
data_to_draw = np.zeros(shape = (1, 50))

def animate(i):
    global data_to_draw
    data_to_draw = np.vstack((data_to_draw, data[i]))
    if data_to_draw.shape[0] > 5:
        data_to_draw = data_to_draw[1:]

    ax.cla()
    sns.heatmap(ax = ax, data = data_to_draw, cmap = "coolwarm", cbar_ax = cbar_ax)

def pandas_dynamic_plot():
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

    g = sns.relplot(x='x', y='y', hue='category', data=df, kind='scatter')

    for i in range(100):
        df['y'] = np.random.randn(100)
        g.data = df
        g.draw()
        plt.pause(0.1)

def python_sucks():
    '''
        Googling shit for half an hour and this is the only
        one that fucking works. A lot of it was dominated 
        by IPython and Jupyter. Those are not frameworks, but
        a lot of people are learning on them I suppose.
    '''
    x = np.linspace(1, 1000, 5000)
    y = np.random.randint(1, 1000, 5000)
    
    # enable interactive mode
    plt.ion()
    
    # creating subplot and figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y)
    
    # setting labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Updating plot...")
    
    # looping
    for i in range(50):
    
        # updating the value of x and y
        line1.set_xdata(x*i)
        line1.set_ydata(y)
    
        # re-drawing the figure
        fig.canvas.draw()
        
        # to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(0.5)