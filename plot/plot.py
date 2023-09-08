# Plotting
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import collections
import seaborn as sns
import torch

import pandas as pd
import time



class PlotInteractive():
    x_lim: any
    y_lim: any

    lines: dict = {}

    def __init__(self, x, y):
        x = x
        y = y
        
        # enable interactive mode
        self.ionContext = plt.ion()
        
        # creating subplot and figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.autoscale(enable=True)

        #self.line1, = self.ax.plot(x, y)

        self.x_lim=(torch.tensor(0.), torch.tensor(0.))
        self.y_lim=(torch.tensor(0.), torch.tensor(0.))
        
        # setting labels
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Training...")
    
    def get_line(self, line_id, x, y):
        if line_id in self.lines:
            return self.lines[line_id]
        else:
            self.lines[line_id], = self.ax.plot(x, y, label=line_id)
            self.ax.legend()
            return self.lines[line_id]

    
    def extents(self, x, y, zoom=True):
        x_lim = (torch.min(x), torch.max(x))
        y_lim = (torch.min(y), torch.max(y))

        x_min = torch.min(self.x_lim[0], x_lim[0])
        x_max = torch.max(self.x_lim[1], x_lim[1])
        y_min = torch.min(self.y_lim[0], y_lim[0])
        y_max = torch.max(self.y_lim[1], y_lim[1])
        
        self.x_lim = (x_min, x_max)
        self.y_lim = (y_min, y_max)

        if(zoom):
            self.shrink_zoom(x_lim, y_lim)
    
    def shrink_zoom(self, xlim, ylim, percent=.75):
        x_r = xlim[1] - xlim[0]
        y_r = ylim[1] - ylim[0]

        self.x_r = self.x_lim[1] - self.x_lim[0]
        self.y_r = self.y_lim[1] - self.y_lim[0]

        # Probably won't work with more advanced sets
        if(x_r/self.x_r < percent):
            self.x_lim = (self.x_lim[0] * percent, self.x_lim[1] * percent)
        
        if(y_r/self.y_r < percent):
            self.y_lim = (self.y_lim[0] * percent, self.y_lim[1] * percent)

    def update(self, line_id, x, y):

        self.extents(x,y)
        
        line = self.get_line(line_id, x, y)
        line.set_xdata(x)
        line.set_ydata(y)
    
        # re-drawing the figure
        self.fig.canvas.draw()

        # fit
        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)
        # to flush the GUI events
        self.fig.canvas.flush_events()
    
    def fin(self, bias):
        plt.title("Trained ({:.2f})".format(bias))
        plt.show(block=True)

