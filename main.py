import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam
import dataset
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    model.train()
    total_loss = 0

    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0))
        if isinstance(hidden, tuple):  # For LSTM, hidden is a tuple (h, c)
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        
        # Flatten the outputs and targets
        outputs = outputs.view(-1, model.input_dim)
        targets = targets.view(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    trn_loss = total_loss / len(trn_loader)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0))
            if isinstance(hidden, tuple):  # For LSTM, hidden is a tuple (h, c)
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = hidden.to(device)
            
            # Forward pass
            outputs, hidden = model(inputs, hidden)
            
            # Flatten the outputs and targets
            outputs = outputs.view(-1, model.input_dim)
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    val_loss = total_loss / len(val_loader)
    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        2) model
        3) optimizer
        4) cost function: use torch.nn.CrossEntropyLoss

    """

    # config
    input_file = "./shakespeare_train.txt"
    shakespeare_dataset = dataset.Shakespeare(input_file)
    batch_size = 256
    lr = 0.001
    num_epochs = 30
    
   # Split dataset into training and validation sets
    dataset_size = len(shakespeare_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))
    np.random.shuffle(indices)
    
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(shakespeare_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(shakespeare_dataset, batch_size=batch_size, sampler=val_sampler)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Choose model type: CharRNN or CharLSTM
    # model = CharLSTM(input_dim=shakespeare_dataset.X.shape[-1], hidden_dim=64, num_layers=3, drop_prob=0.3)
    model = CharRNN(input_dim=shakespeare_dataset.X.shape[-1], hidden_dim=64, num_layers=3, drop_prob=0.3)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    trn_loss_list, val_loss_list = [], []
    
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, device, criterion, optimizer)
        trn_loss_list.append(train_loss)
        
        val_loss = validate(model, val_loader, device, criterion)
        val_loss_list.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # save model
    model_name = 'CharRNN'
    torch.save(model, f'{model_name}.pt')
   
    # visualize
    plt.figure(figsize=(8,6))
    sns.lineplot(trn_loss_list, marker='o', color='blue', label='training')
    sns.lineplot(val_loss_list, marker='o', color='red', label='validation')
    plt.legend()
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(f'train_{model_name}.png')
    plt.show()
    

if __name__ == '__main__':
    main()
