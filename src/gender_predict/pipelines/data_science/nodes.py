"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.6
"""
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from string import ascii_lowercase


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le

import torch
import torch.nn as nn
from torch.autograd import Variable


def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets

    Args:
        model_input_table (pd.DataFrame): Data containing features and target
        parameters (Dict): Parameters defined in yml

    Returns:
        Tuple: Splitted data
    """
    # features to predict label
    X = model_input_table[parameters["features"]].to_numpy()
    # label to predict
    y = le().fit_transform(model_input_table[parameters["target"]])
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = parameters["test_size"], random_state = parameters["random_state"], stratify = y
    )
    return X_train, X_test, y_train, y_test


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        new_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = nn.LeakyReLU(0.02)(output)
        output = self.softmax(output)
        return output, new_hidden

    def init_hidden(self):
        return Variable(torch.zeros(1,self.hidden_size))


def init_device(parameters: Dict) -> torch.device:
    """instantiate device

    Args:
        parameters (Dict): yaml configuration

    Returns:
        torch.device: use GPU if available or CPU
    """
    return torch.device(parameters['gpu'] if torch.cuda.is_available() else parameters['cpu'])


def init_neural_network(device: torch.device, parameters: Dict) -> RNN:
    """initialize neural network

    Args:
        device (torch.device): GPU or CPU
        parameters (Dict): yaml configuration

    Returns:
        RNN: Recurrent Neural Network
    """
    return RNN(len(ascii_lowercase), parameters["n_hiddens"], parameters["n_classes"]).to(device)


#def __train(name_tensor,class_tensor,device):
#    rnn.zero_grad()
#    hidden=rnn.init_hidden()
#    hidden=hidden.to(device)
#    for i in range(name_tensor.size()[0]):
#        output,hidden=rnn(name_tensor[i],hidden)
#
#    loss=criterion(output,class_tensor)
#    loss.backward()
#    optimizer.step()
#    return output,loss.item()
#
#
#def train_neural_network(X_train: np.array, y_train: np.array, rnn: RNN, device: torch.device, parameters: Dict):
#    """train neural network and evaluate the model
#
#    Args:
#        X_train (np.array): [description]
#        rnn (RNN): [description]
#        device (torch.device): [description]
#        parameters (Dict): [description]
#    """
#    # init list to store values
#    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
#
#    # training loops
#    for epoch in range(parameters["epochs"]):
#
#        # init training loss
#        train_loss = parameters["train_loss"]
#
#        # random shuffle people
#        np.random.shuffle(np.arange(len(X_train)))
#
#        # iterate through names with idx
#        for i,idx in enumerate(np.arange(len(X_train))):
#
#            # get name with tensor representation and index position
#            name_tensor, class_tensor = get_data_pair(X_train, y_train, idx)
#
#            # store name representation to tensor
#            name_tensor = name_tensor.to(device)
#
#            # store class value to tensor
#            class_tensor = class_tensor.to(device)
#
#            # train to get output and loss
#            output, loss = train(name_tensor, class_tensor, device)
#
#            # append training loss
#            train_loss += loss
#            # if finish training step by name is over reinitialize train loss score
#            if(i % parameters["print_steps"] == 0 and i > 0 and i % parameters["print_all_steps"] != 0):
#                print("Iter :", i, "at epoch: ", epoch + 1, "/ ", parameters["epochs"], " Train Loss : ", train_loss / parameters["print_steps"])
#                train_loss = parameters["train_loss"]
#
#            elif(i % parameters["print_all_steps"] == 0 and i > 0):
#                with torch.no_grad():
#                    train_acc = evaluate(X_train, y_train, device)[1] #Calculate acc for all
#                    train_acc_list.append(train_acc)
#                    train_loss_list.append(train_loss/parameters["print_steps"]) #calculate loss after "print_steps" training samples and get mean
#                    val_loss,val_acc=evaluate(X_val,y_val,device)
#                    val_loss_list.append(val_loss)
#                    val_acc_list.append(val_acc)
#                    print("Iter: ",i,"at epoch: ",epoch+1,"/",parameters["epochs"]," train loss :",train_loss/parameters["print_steps"],"Train acc: ",train_acc," val loss: ",val_loss," val acc: ",val_acc)
#                    train_loss=0
#
#
#
#def evaluate(X,y,device):
#    loss=0
#    correct=0
#    for i in range(len(X)):
#        name_tensor,class_tensor=get_data_pair(X,y,i)
#        name_tensor=name_tensor.to(device)
#        class_tensor=class_tensor.to(device)
#        hidden=rnn.init_hidden().to(device)
#    for j in range(name_tensor.size()[0]):
#        predicted,hidden=rnn(name_tensor[j],hidden)
#        loss+=criterion(predicted,class_tensor)
#        idx_predicted=torch.max(predicted.data,1)[1]
#        correct+=(idx_predicted==class_tensor).sum()
#    return loss/len(X),correct.item()/len(X)*100