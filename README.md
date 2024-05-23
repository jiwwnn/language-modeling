# language-modeling
### Average loss values of training and validation datasets 
- Loss of RNN decreases more than that of LSTM for both training and validation
  - [RNN] training loss : 1.8593 / validation loss : 1.8632
  <img src="https://github.com/jiwwnn/language-modeling/assets/134251617/8d51483f-7c37-43d1-8908-9da5a1ffead9" width='500'>

  - [LSTM] training loss : 1.6926 / validation loss : 1.7205
  <img src="https://github.com/jiwwnn/language-modeling/assets/134251617/ea25d946-11bc-4a12-b5b1-a291bcc15353" width='500'>

### Comparing the language generation performances in terms of loss values for validation dataset
  - LSTM is superior than RNN in terms of loss values for validation dataset.
  - **Comparison of val loss (epoch 30) : [RNN] 1.8632  > [LSTM] 1.7205 (-0.1427)**
  - This result could derive from the improvement of long-term dependencies in LSTM which shows still working better in character level than RNN

### Generating at least 100 length of 5 different samples from seed characters
- Best model : LSTM
- Seed characters : speak, citizen, child, heaven, member
- Results
  - 
