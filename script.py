import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#Sudokus to load
load_number = 1000


quizzes = np.zeros((load_number+1, 81), np.int32)
solutions = np.zeros((load_number+1, 81), np.int32)
for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s
    if i == load_number:
        break
quizzes = quizzes.reshape((-1, 9, 9))
solutions = solutions.reshape((-1, 9, 9))

dataset = {}
dataset['quizzes'] = quizzes
dataset['solutions'] = solutions

class SudokuDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, sudoku_list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sudoku_list = sudoku_list
        self.transform = transform

    def __len__(self):
        return len(self.sudoku_list['quizzes'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        quiz = self.sudoku_list['quizzes'][idx]
        solution = self.sudoku_list['solutions'][idx]
        sample = {'quizzes': quiz, 'solutions': solution}

        if self.transform:
            sample = self.transform(sample)

        return sample

dataset = SudokuDataset(dataset)

''' Config '''
batch_size = 5
shuffle = True

#Dataloader
train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=  batch_size,
        shuffle=  True,
        drop_last=False)

train_iterator = train_loader.__iter__()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
         
        self.placement_net = nn.Sequential(
            nn.Linear( 81, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 81),
            #nn.Softmax(dim=1)
        )

        self.number_net = nn.Sequential(
            nn.Linear( 82, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 81),
            nn.ReLU(inplace=True),
            nn.Linear( 81, 9),
            #nn.Softmax(dim=1)
        )

    def forward(self, batch):
        quiz = batch['quizzes'] # Dim B x 9 x 9
        B = quiz.shape[0]
        quiz = quiz.reshape(B, 81) # Dim B x 81


        placement_prob = self.placement_net(quiz.float())
        _ , placement_guess = torch.max(placement_prob, 1)
        placement = placement_guess.reshape(-1,1)
        number_prob = self.number_net( torch.cat((quiz , placement.int()), dim=1 ).float() )
        _ , number_guess = torch.max(number_prob, 1)

        #placement_values , placement_guess = torch.max(placement_prob, 1)
        #placement =  placement_values.reshape(-1,1)       
        #number_prob = self.number_net( torch.cat((quiz , placement.int()), dim=1 ).float() )
        #number_values , number_guess = torch.max(number_prob, 1)
        
        return placement_guess, number_prob  #  placement_values, placement_guess, placement_prob, number_values, number_guess

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)




def loss_function(prediction, solution):
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(prediction, solution)
    return loss

def specific_loss_function(placement_guess, number_guess, solution):
    loss_func = nn.CrossEntropyLoss()
    B = solution.shape[0]
    solution = solution.reshape(-1)
    solution = solution[torch.arange(0,B),placement_guess]
    
    solution[:,placement_guess]
    loss = loss_func(number_guess, solution)
    
    return loss

for i in range(1000):
    optimizer.zero_grad()
    batch = train_iterator.next()
    placement_guess, number_guess  = model.forward(batch)
    #prediction = prediction.reshape(-1,9)
     
    loss = specific_loss_function(placement_guess, number_guess , batch['solutions'].long()-1)
    _ , prediction0 = torch.max(prediction, 1)
    loss.backward()
    optimizer.step()
    print(loss)
    #torch.save(model.state_dict, )