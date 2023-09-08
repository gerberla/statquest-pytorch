import torch
import torch.nn as nn
import torch.nn.functional as af
from torch.optim import SGD as sgd

import numpy

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

from plot.plot import PlotInteractive

class SimpleNN_train(nn.Module):

    def __init__(self):
        super().__init__()

        # Top
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        # Bottom
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def train(self, inputs, labels, test_inputs, learning_rate=0.1, epochs=100):
        optimizer = sgd(self.parameters(), learning_rate)
        print("Final bias, before optimization: " + str(self.final_bias.data) + "\n")

        plot = PlotInteractive(inputs, [0] * len(inputs))

        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                input_i = inputs[i]
                label_i = labels[i]

                output_i = self(input_i)

                # Calculate Square Residual b/w output and known value
                loss = (output_i - label_i)**2

                loss.backward()
                total_loss += float(loss)
            if(total_loss < 0.0001):
                print("Num steps: " + str(epoch))
                break
            out = self(inputs).detach()
            plot.update('train', inputs, out)
            plot.update('test', test_inputs, self(test_inputs).detach())
            plt.pause(.1) # TODO Move out of this function

            optimizer.step()
            optimizer.zero_grad()

            print("Step: " + str(epoch) + " Final Bias: " + str(self.final_bias.data) + "\n")
        print("Final Bias, after optimization: " + str(self.final_bias.data))
        plot.fin(self.final_bias.data)

    def forward(self, input):
        # Top
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = af.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        
        # Bottom
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = af.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias

        output = af.relu(input_to_final_relu)

        return output
        