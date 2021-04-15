import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import learn2learn as l2l


class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    mnist = MNIST(root="/tmp/mnist", train=True)

    mnist = l2l.data.MetaDataset(mnist)
    task_generator = l2l.data.TaskGenerator(mnist,
                                            ways=3,
                                            classes=[0, 1, 4, 6, 8, 9],
                                            tasks=10)
    model = Net()
    maml = l2l.algorithms.MAML(model, lr=1e-3, first_order=False)
    opt = optim.Adam(maml.parameters(), lr=4e-3)

    for iteration in range(num_iterations):
        learner = maml.clone()  # Creates a clone of model
        adaptation_task = task_generator.sample(shots=1)

        # Fast adapt
        # for the same task
        for step in range(adaptation_steps):
            error = compute_loss(adaptation_task)
            learner.adapt(error)

        # Compute evaluation loss - on the same task?
        evaluation_task = task_generator.sample(shots=1,
                                                task=adaptation_task.sampled_task)
        evaluation_error = compute_loss(evaluation_task)

        # Meta-update the model parameters
        opt.zero_grad()
        evaluation_error.backward()
        opt.step()
