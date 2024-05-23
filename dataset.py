import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load input file 
        with open(input_file, 'rb') as f:
            # preprocessing
            sentences = []
            for sentence in f:
                sentence = sentence.strip() # remove \r, \n 
                sentence = sentence.lower() # lowercase letters
                sentence = sentence.decode('ascii', 'ignore') 
                if len(sentence) > 0:
                    sentences.append(sentence)
            f.close()
        
        # Construct character dictionary {index:character}
        total_data = ''.join(sentences)
        self.char2idx = {char: idx for idx, char in enumerate(sorted(set(total_data)))}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        # Make list of character indices using the dictionary
        seq_length = 30
        n_samples = int(np.floor(len(total_data) / seq_length))
       
        train_X = []
        train_y = []
        
        for i in range(n_samples):
            # 0:30 -> 30:60 -> ...
            X_sample = total_data[i*seq_length : (i+1) * seq_length]
            # encoding (char -> idx)
            X_encoded = [self.char2idx[c] for c in X_sample]
            train_X.append(X_encoded)
            
            # shift +1 to right direction
            y_sample = total_data[i*seq_length+1 : (i+1)*seq_length+1]
            # encoding (char -> idx)
            y_encoded = [self.char2idx[c] for c in y_sample]
            train_y.append(y_encoded)
        
        self.X = torch.LongTensor(train_X)  
        self.X = F.one_hot(self.X).float()
        self.Y = torch.LongTensor(train_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input = self.X[idx]
        target = self.Y[idx]
        return input, target

if __name__ == '__main__':
    # Test the implementation
    input_file = "./shakespeare_train.txt"  # Change this to your actual input file
    shakespeare_dataset = Shakespeare(input_file)

    # Check the length of the dataset
    print("Length of dataset:", len(shakespeare_dataset))

    # Get a sample item from the dataset
    sample_input, sample_target = shakespeare_dataset[0]
    print("Sample input:", sample_input)
    print((sample_input).shape) # [30, 36]
    print("Sample target:", sample_target)
