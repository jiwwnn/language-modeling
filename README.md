# language-modeling
### Average loss values of training and validation datasets 
  - RNN : Loss of RNN decreases rapidly than that of LSTM 
![train_CharRNN](img src="https://github.com/jiwwnn/language-modeling/assets/134251617/28661317-a7b6-4919-bcb5-e3dc38034128" width='500')
  - LSTM : Loss decreases gradually compared to RNN, and the total amount of loss reduction is less than that of RNN 
![train_CharLSTM](img src="https://github.com/jiwwnn/language-modeling/assets/134251617/0f14555f-5d01-4d09-8a51-c120e0ad11dd" width='500')

### Comparing the language generation performances in terms of loss values for validation dataset
  - RNN is superior than LSTM in terms of loss values for validation dataset.   - [Comparison of val loss (epoch 30)] (RNN) 2.1422  < (LSTM) 2.4303
