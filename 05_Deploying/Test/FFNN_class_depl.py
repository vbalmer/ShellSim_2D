'''
NOTE: This is a copy of the file FFNN_class. Do not change it's contents unless you also want to 
change it in the original class
'''



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import numpy as np
import wandb
from torch import randn, random
import math

# def input_mapping(x:torch.Tensor, inp:dict):
#     # https://arxiv.org/pdf/2006.10739
#     if not inp['fourier_mapping']:
#         return x
#     else:
#         # gen = torch.Generator()
#         # gen.manual_seed(42)
#         B = randn((inp['input_size'], inp['num_samples']))  #, gen)
#         x_proj = torch.matmul((2.*(torch.Tensor([math.pi]))*x), B)
#         return torch.concat([np.sin(x_proj), np.cos(x_proj)], axis=-1)



class FourierMapping(nn.Module):
    def __init__(self, input_dim, num_samples=5):
        super(FourierMapping, self).__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = input_dim * (2 * num_samples + 1)
        self.freqs = nn.Parameter(torch.Tensor(1, input_dim, num_samples), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.freqs, -torch.pi, torch.pi)

    def forward(self, x):
        fourier_features = torch.cat([
            torch.sin(x.unsqueeze(-1) * self.freqs),
            torch.cos(x.unsqueeze(-1) * self.freqs)
        ], dim=-1).reshape(x.size(0), self.output_dim - self.input_dim)
        return torch.cat([x, fourier_features], dim=-1)


# Define the neural network class
class FFNN(nn.Module):
    def __init__(self, inp: dict):
        super(FFNN, self).__init__()
        if inp['fourier_mapping']:
            self.fourier_mapping = FourierMapping(inp['input_size'])

        layers = []
        for i in range(len(inp['hidden_layers']) - 1):
            layers.append(nn.Dropout(inp['dropout_rate']))
            layers.append(nn.Linear(inp['hidden_layers'][i], inp['hidden_layers'][i + 1]))
            layers.append(inp['activation']())
        
        if inp['fourier_mapping']:
            inp_size = self.fourier_mapping.output_dim
        else: 
            inp_size = inp['input_size']

        self.layers = nn.Sequential(
            nn.Linear(inp_size, inp['hidden_layers'][0]),
            inp['activation'](),
            *layers,
            nn.Linear(inp['hidden_layers'][-1], inp['out_size'])
        )

    def forward(self, x):
        out = self.fourier_mapping(x)
        for layer in self.layers:
            out = layer(out)
        return out

    def training(data_loader: DataLoader, model: nn.Module, inp: dict):
        '''
        data_loader     (torch dataloader)      contains features, labels for training in structured manner
        model           (nn.Module)             model architecture as built in previous section
        inp             (dict)                  Input parameters such as learning rate, num_epochs
        '''
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=inp['learning_rate'])

        def lr_lambda(current_step: int, min_factor=0.01):
            warmup_steps = inp['num_epochs'] * 0.1
            if current_step < warmup_steps:
                return (1 - min_factor) * (current_step / warmup_steps) ** 2 + min_factor
            progress = (current_step - warmup_steps) / (inp['num_epochs'] - warmup_steps)
            return (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * progress)) + min_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(inp['num_epochs']):
            model.train()
            total_loss = 0
            for i, (features, labels) in enumerate(data_loader):
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            scheduler.step()
            wandb.log({"loss": loss})
                
            if (epoch + 1) % 10 == 0:
                num_epochs = inp['num_epochs']
                current_lr = scheduler.get_last_lr()[0]
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        return model

    def evaluate(model, test_loader: DataLoader, inp:dict):
        # Test the model (Make into own function?)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss = 0
            i = 0
            for features, labels in test_loader:
                outputs = model(features)
                loss = np.mean((outputs.numpy() - labels.numpy()) ** 2)
                val_loss += loss.item()
                i = i+1
                val_avg_loss = val_loss/len(list(test_loader))
                # print(len(test_loader))
                print(f'Validation loss of the model on the evaluation data:', val_avg_loss)
                wandb.log({"val_avg_loss": val_avg_loss})


            # with torch.no_grad():
                # val_outputs = model(self.X_val)
                # val_loss = np.mean((val_outputs.numpy() - self.y_val.numpy()) ** 2)
               
        return model
