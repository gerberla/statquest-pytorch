import torch
import matplotlib.pyplot as plt
import seaborn as sns
import RunSpace

from models.SimpleNN import SimpleNN
from models.SimpleNN_train import SimpleNN_train
from tensors.LinSpace import LinSpace

from garbage.plotting import *


def doses_test(model, ls: LinSpace):
    input_doses = torch.linspace(ls.start, ls.end, ls.step)
    print(input_doses)

    #efficacy = torch.tensor()

    output_value = model(input_doses)

    print(output_value)

    sns.set(style="whitegrid")
    sns.lineplot(x=input_doses,
                     y=output_value,
                     color='green',
                     linewidth=2.5)
    
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')

    plt.show()

def doses_train(model: SimpleNN_train, ls: LinSpace):
    '''
        Use matplotlib to plot training and testing data for each
        model regression
    '''
    input_doses = torch.linspace(ls.start, ls.end, ls.step)

    # Train
    input = torch.tensor([0.,.5,1.])
    labels = torch.tensor([0,1,0])

    model.train(input, labels, input_doses)
    output_value = model(input_doses)

    print(output_value)


def shit_try():
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (10, 8))
    ani = FuncAnimation(fig = fig, func = animate, frames = 100, interval = 100)

    plt.show()

if __name__ == "__main__":
    doses = LinSpace(0,1,21)
    #doses_test(SimpleNN(), doses)
    doses_train(SimpleNN_train(), doses)

    
