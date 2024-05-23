import torch
import numpy as np
import dataset
from model import CharRNN, CharLSTM
import torch.nn.functional as F

def generate(model, seed_characters, temperature, max_length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: temperature parameter for controlling the randomness of the generated text
        max_length: maximum length of generated text

    Returns:
        samples: generated characters
    """

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize hidden state
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):  # For LSTM, hidden is a tuple (h, c)
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)

    # Encode the seed characters
    seed_indices = [char2idx[c] for c in seed_characters]
    input_seq = torch.tensor(seed_indices, dtype=torch.long).unsqueeze(0).to(device)
    input_seq = F.one_hot(input_seq, num_classes=len(char2idx)).float()

    generated_chars = []
    generated_chars.extend(seed_characters)

    # Generate characters
    for _ in range(num_chars):
        output, hidden = model(input_seq, hidden)
        
        # Take the last output from the sequence
        output = output[:, -1, :]  # [1, len(char2idx)]
        
        # Apply temperature and softmax to get probabilities
        output = output / temperature
        probabilities = F.softmax(output, dim=-1).cpu().detach().numpy().squeeze()
        
        # Sample from the distribution
        char_index = np.random.choice(len(char2idx), p=probabilities)
        
        # Add the generated character to the sequence
        generated_chars.append(idx2char[char_index])
        
        # Update input_seq with the newly generated character
        input_seq = torch.tensor([[char_index]], dtype=torch.long).to(device)
        input_seq = F.one_hot(input_seq, num_classes=len(char2idx)).float()

    return ''.join(generated_chars)

if __name__ == '__main__':
    # Example usage
    input_file = "./shakespeare_train.txt"  
    shakespeare_dataset = dataset.Shakespeare(input_file)
    
    char2idx = shakespeare_dataset.char2idx
    idx2char = shakespeare_dataset.idx2char
    
    # model = CharLSTM(input_dim=len(char2idx), hidden_dim=64, num_layers=3, drop_prob=0.3)
    model = CharRNN(input_dim=len(char2idx), hidden_dim=64, num_layers=3, drop_prob=0.3)
    model = torch.load('./CharRNN.pt')
    
    model.eval()
    
    seed = "to be or not to be"
    temperature = 1.0
    num_chars = 100
    
    generated_text = generate(model, seed, temperature, num_chars)
    print(generated_text)
    
    