
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from math import ceil
from tqdm import tqdm
import time
from collections import OrderedDict
import copy
import time
from .hn_utils import make_dice
import numpy as np

# ji = MulticlassJaccardIndex(num_classes=12)

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_optimizer(net, name, lr):
    """'adam' -> Adam(lr); 'sgd' -> SGD(lr, momentum=0.9). These are the two
    optimisers used in the original HYDRA-FL runs."""
    name = (name or 'adam').lower()
    if name == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    if name == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer '{name}' (use 'adam' or 'sgd')")


def calculate_metrics(pred, target, metric_fn, device):
    pred = torch.argmax(pred, dim=1)

    return metric_fn(pred, target).to(device)

def load_data(trainset, testset):
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples

def flush_memory():
    torch.cuda.empty_cache()

def train_model(net, trainloader, epochs, device, wait_time, deadline=None):

    """
    Trains a neural network model on a given dataset using SGD optimizer with Cross Entropy Loss criterion.
    Args:
        net: neural network model
        trainloader: PyTorch DataLoader object for training dataset
        epochs: number of epochs to train the model
        deadline: optional deadline time for training

    Returns:
        trained model with the difference between trained model and the received model
    """
    x = deepcopy(net)
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Set the model to training mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            labels = labels.view(-1).long()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
        time.sleep(wait_time)
        if deadline:
            if time.time()>=deadline:
                print('Deadline occured!')
                break

    #evaluate model on trainset
    # eval_loss, eval_accuracy = test_model(net, trainloader, device)
    # eval_result = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}
    # print("Eval results: ", eval_result)

    # Calculate the difference between the trained model and the received model
    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    return net

def train_scaffold(net, server_c, client_c, trainloader, epochs, device, wait_time, deadline=None):
    """
    Trains a given neural network using the Scaffold algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model

    """
    x = deepcopy(net)
    if client_c is None:
        client_c = deepcopy(server_c)
    criterion = torch.nn.CrossEntropyLoss()
    lr = 0.001

    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1).long()
            loss = criterion(net(images), labels)

            #Compute (full-batch) gradient of loss with respect to net's parameters
            grads = torch.autograd.grad(loss,net.parameters())

            #Update y's parameters using gradients, client_c and server_c [Algorithm line no:10]

            for param,grad,s_c,c_c in zip(net.parameters(),grads,server_c,client_c):
                s_c, c_c = s_c.to(device), c_c.to(device)
                param.data = param.data - lr * (grad.data + (s_c.data - c_c.data))
        time.sleep(wait_time)
        if deadline:
            if time.time()>=deadline:
                print('Deadline occured!')
                break

    delta_c = [torch.zeros_like(param) for param in net.parameters()]
    new_client_c = deepcopy(delta_c)

    for param_net, param_x in zip(net.parameters(), x.parameters()):
        param_net.data = param_net.data - param_x.data

    a = (ceil(len(trainloader.dataset) / trainloader.batch_size) * epochs * lr)
    for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, net.parameters()):
        c_l = c_l.to(device)
        c_g = c_g.to(device)
        n_c.data += c_l.data - c_g.data - diff.data / a

    #Calculate delta_c which equals to new_client_c-client_c
    for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
        d_c = d_c.to(device)
        c_l = c_l.to(device)
        d_c.data.add_(n_c_l.data - c_l.data)


    return net, delta_c, new_client_c

def train_fedavg(net, trainloader, valloader, epochs, device, lr, optimizer_name=None, verbose=True):
    """
    Trains a given neural network using the Federated Averaging (FedAvg) algorithm.

    Args:
        net: A PyTorch neural network model
        trainloader: A PyTorch DataLoader containing the training dataset

    Returns:
        A trained PyTorch neural network model
    """
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # Original runs: Adam for FedAvg clients.
    optimizer = _make_optimizer(net, optimizer_name or 'adam', lr)
    D = make_dice(11, device)

    train_dice = 0.0
    val_dice = 0.0

    val_dice_values = []

    print("Getting val metrics before training:")
    net.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()

            val_dice = calculate_metrics(outputs, labels, D, device)
            val_dice_values.append(val_dice.item())

    avg_val_dice = np.mean(val_dice_values)
    print(f"Before Training, Val Dice: {avg_val_dice}")

    # Train the model for the specified number of epochs
    for epoch in tqdm(range(epochs)):
        net.train()
        train_dice_scores = []
        val_dice_scores = []

        # if epoch == 2:
        #     #unfreeze all the layers of the model
        #     for param in net.parameters():
        #         param.requires_grad = True

        for images, labels in trainloader:
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()
            
            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()

            train_dice = calculate_metrics(outputs, labels, D, device)
            train_dice_scores.append(train_dice.item())

        net.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                labels = labels.long()

                val_dice = calculate_metrics(outputs, labels, D, device)
                val_dice_scores.append(val_dice.item())

        avg_train_dice = np.mean(train_dice_scores)
        avg_val_dice = np.mean(val_dice_scores)

        print(f"Epoch: {epoch+1}, Train Dice: {avg_train_dice}")
        print(f"Epoch {epoch+1}, Val Dice: {avg_val_dice}")

    return net


def train_fedprox(net, trainloader, epochs, device, mu, wait_time, deadline = None):
    """
    Trains a given neural network using the Federated Proximal (Fedprox) algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    # create a copy of the initial model
    init_model = copy.deepcopy(net)
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Set model to train mode
    net.train()

    # Train the model for the specified number of epochs
    for _ in tqdm(range(epochs)):
        for images, labels in trainloader:
            Prox_loss = 0.0
            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Compute the loss
            #convert labels to 1d and long
            labels = labels.view(-1).long()
            loss = criterion(outputs, labels)
            for paramA, paramB in zip(init_model.parameters(), net.parameters()):
                Prox_loss += torch.square(torch.norm((paramA.detach() - paramB)))
            Prox_loss = (mu/2)*Prox_loss

            # Backward pass
            (loss + Prox_loss).backward()

            # Update model parameters
            optimizer.step()
        time.sleep(wait_time)
        if deadline:
            if time.time()>=deadline:
                print('Deadline occured!')
                break

    # Return the trained model
    return net

def test_model(net, testloader, device): ## make changes here for classification
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader) :
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            # labels = labels.view(-1).long()
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    iou = ji(outputs, labels)
    return loss, accuracy, iou



# trainloader, testloader, num_examples = load_data()
# model = Net().to(DEVICE)
# torch.save(model.state_dict(), "test_state_dict.pth")
# model.load_state_dict(torch.load('trained_model.pt'))
# train_model(model, trainloader, 3)
# print(test_model(model, testloader))
# # torch.save(model.state_dict(), 'trained_model.pt')

def fedadam(initial_state_dict, trained_state_dict):

    #### Calculate the difference between the initial state dict and the trained state dict
    delta_y = OrderedDict()
    for key in initial_state_dict.keys():
        delta_y[key] = trained_state_dict[key] - initial_state_dict[key].to(trained_state_dict[key].device)

    return delta_y

def train_feddyn(net, trainloader, epochs, device, prev_grads, wait_time, deadline=None):
    """
    Trains a given neural network using the FedDyn algorithm.
    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    epochs: An integer specifying the number of training epochs
    deadline: An optional deadline (in seconds) for the training process

    Returns:
    A trained PyTorch neural network model
    """
    x = deepcopy(net)
    # prev_grads = None

    if prev_grads is not None:
        prev_grads = prev_grads.to(device)
    else:
        for param in net.parameters():
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1))
                prev_grads.to(device)
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
                prev_grads.to(device)

    criterion = torch.nn.CrossEntropyLoss()
   
    lr = 0.1
    alpha = 0.01

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        inputs,labels = next(iter(trainloader))
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        output = net(inputs)
        #convert labels to 1d and long
        labels = labels.view(-1).long()
        loss = criterion(output, labels) #Calculate the loss with respect to y's output and labels

        #Dynamic Regularisation
        lin_penalty = 0.0
        curr_params = None
        for param in net.parameters():
            if not isinstance(curr_params, torch.Tensor):
                curr_params = param.view(-1)
            else:
                curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

        lin_penalty = torch.sum(curr_params * prev_grads)
        loss -= lin_penalty

        quad_penalty = 0.0
        for y, z in zip(net.parameters(), x.parameters()):
            quad_penalty += torch.nn.functional.mse_loss(y.data, z.data, reduction='sum')

        loss += (alpha/2) * quad_penalty
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=1) # Clip gradients
        optimizer.step()

        # gradients = torch.autograd.grad(loss,net.parameters())

        # for param, grad in zip(net.parameters(),gradients):
        #     param.data -= lr * grad.data
        time.sleep(wait_time)
        if deadline:
                if time.time()>=deadline:
                    print('Deadline occured!')
                    break
    #Calculate the difference between updated model (y) and the received model (x)
    delta = None
    for y, z in zip(net.parameters(), x.parameters()):
        if not isinstance(delta, torch.Tensor):
            delta = torch.sub(y.data.view(-1), z.data.view(-1))
        else:
            delta = torch.cat((delta, torch.sub(y.data.view(-1), z.data.view(-1))),dim=0)

    #Update prev_grads using delta which is scaled by alpha
    prev_grads = torch.sub(prev_grads, delta, alpha = alpha)
    return net, prev_grads

def train_h_feddyn(net, trainloader, valloader, epochs, device, prev_grads, lr, alpha=0.01):
    """
    Trains a given neural network using the FedDyn algorithm.

    Args:
    net: A PyTorch neural network model
    trainloader: A PyTorch DataLoader containing the training dataset
    valloader: A PyTorch DataLoader containing the validation dataset
    epochs: An integer specifying the number of training epochs
    prev_grads: The previous gradients of the model
    lr: The learning rate of the optimizer
    alpha: The regularization parameter

    Returns:
    A trained PyTorch neural network model and the updated previous gradients
    """
    x = deepcopy(net)

    if prev_grads is not None:
        prev_grads = prev_grads.to(device)
    else:
        for param in net.parameters():
            if not isinstance(prev_grads, torch.Tensor):
                prev_grads = torch.zeros_like(param.view(-1))
                prev_grads.to(device)
            else:
                prev_grads = torch.cat((prev_grads, torch.zeros_like(param.view(-1))), dim=0)
                prev_grads.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    D = make_dice(11, device)

    train_dice = 0.0
    val_dice = 0.0

    val_dice_values = []

    print("Getting val metrics before training:")
    net.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()

            val_dice = calculate_metrics(outputs, labels, D, device)
            val_dice_values.append(val_dice.item())

    avg_val_dice = np.mean(val_dice_values)
    print(f"Before Training, Val Dice: {avg_val_dice}")

    for epoch in tqdm(range(epochs)):
        net.train()
        train_dice_scores = []
        val_dice_scores = []

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()

            loss = criterion(outputs, labels)

            # Dynamic Regularisation
            lin_penalty = 0.0
            curr_params = None
            for param in net.parameters():
                if not isinstance(curr_params, torch.Tensor):
                    curr_params = param.view(-1)
                else:
                    curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

            lin_penalty = torch.sum(curr_params * prev_grads)
            loss -= lin_penalty

            quad_penalty = 0.0
            for y, z in zip(net.parameters(), x.parameters()):
                quad_penalty += torch.nn.functional.mse_loss(y.data, z.data, reduction='sum')

            loss += (alpha/2) * quad_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_dice = calculate_metrics(outputs, labels, D, device)
            train_dice_scores.append(train_dice.item())

        net.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                labels = labels.long()

                val_dice = calculate_metrics(outputs, labels, D, device)
                val_dice_scores.append(val_dice.item())

        avg_train_dice = np.mean(train_dice_scores)
        avg_val_dice = np.mean(val_dice_scores)

        print(f"Epoch: {epoch+1}, Train Dice: {avg_train_dice}")
        print(f"Epoch {epoch+1}, Val Dice: {avg_val_dice}")

    # Calculate the difference between updated model (y) and the received model (x)
    delta = None
    for y, z in zip(net.parameters(), x.parameters()):
        if not isinstance(delta, torch.Tensor):
            delta = torch.sub(y.data.view(-1), z.data.view(-1))
        else:
            delta = torch.cat((delta, torch.sub(y.data.view(-1), z.data.view(-1))),dim=0)

    # Update prev_grads using delta which is scaled by alpha
    prev_grads = torch.sub(prev_grads, delta, alpha=alpha)

    return net, prev_grads


def train_h_fedprox(net, trainloader, valloader, epochs, device, mu=0.001, lr=0.001, optimizer_name=None, verbose=True):
    """
    Trains a given neural network using the Federated Proximal (FedProx) algorithm.

    Args:
        net: A PyTorch neural network model
        trainloader: A PyTorch DataLoader containing the training dataset
        valloader: A PyTorch DataLoader containing the validation dataset
        epochs: An integer specifying the number of training epochs
        mu: Proximal term coefficient for FedProx
        lr: Learning rate for the optimizer

    Returns:
        A trained PyTorch neural network model
    """
    # Create a copy of the initial model for FedProx proximal term calculation
    init_model = copy.deepcopy(net)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # Original runs: SGD with momentum 0.9 for FedProx clients.
    optimizer = _make_optimizer(net, optimizer_name or 'sgd', lr)
    D = make_dice(11, device)

    train_dice = 0.0
    val_dice = 0.0

    val_dice_values = []

    print("Getting val metrics before training:")
    net.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()

            val_dice = calculate_metrics(outputs, labels, D, device)
            val_dice_values.append(val_dice.item())

    avg_val_dice = np.mean(val_dice_values)
    print(f"Before Training, Val Dice: {avg_val_dice}")

    # Train the model for the specified number of epochs
    for epoch in tqdm(range(epochs)):
        net.train()
        train_dice_scores = []
        val_dice_scores = []

        for images, labels in trainloader:
            # Initialize the Proximal loss
            prox_loss = 0.0

            # Move data to device (GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            labels = labels.long()

            # Compute the main loss
            loss = criterion(outputs, labels)

            # Compute the Proximal term (FedProx specific)
            for paramA, paramB in zip(init_model.parameters(), net.parameters()):
                prox_loss += torch.square(torch.norm((paramA.detach() - paramB)))
            prox_loss = (mu / 2) * prox_loss

            # Backward pass
            (loss + prox_loss).backward()

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()

            # Calculate train Dice score
            train_dice = calculate_metrics(outputs, labels, D, device)
            train_dice_scores.append(train_dice.item())

        # Validation phase
        net.eval()
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                labels = labels.long()

                val_dice = calculate_metrics(outputs, labels, D, device)
                val_dice_scores.append(val_dice.item())

        avg_train_dice = np.mean(train_dice_scores)
        avg_val_dice = np.mean(val_dice_scores)

        print(f"Epoch: {epoch+1}, Train Dice: {avg_train_dice}")
        print(f"Epoch {epoch+1}, Val Dice: {avg_val_dice}")
        
    return net
