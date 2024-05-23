# language-modeling
### 1. Average loss values of training and validation datasets 
- Loss of RNN decreases more than that of LSTM for both training and validation
  - [RNN] training loss : 1.8593 / validation loss : 1.8632
  <img src="https://github.com/jiwwnn/language-modeling/assets/134251617/8d51483f-7c37-43d1-8908-9da5a1ffead9" width='500'>

  - [LSTM] training loss : 1.6926 / validation loss : 1.7205
  <img src="https://github.com/jiwwnn/language-modeling/assets/134251617/ea25d946-11bc-4a12-b5b1-a291bcc15353" width='500'>

### 2. Comparing the language generation performances in terms of loss values for validation dataset
  - LSTM is superior than RNN in terms of loss values for validation dataset.
  - Comparison of val loss (epoch 30) : [RNN] 1.8632  > [LSTM] 1.7205 (-0.1427)
  - This result could derive from the improvement of long-term dependencies in LSTM which shows still working better in character level than RNN

### 3. Generating at least 100 length of 5 different samples from seed characters
- Best model : LSTM
- Seed characters : speak / citizen / child / heaven / member
- Results
  - (1) speak - speak.menenius:that you one's to death.gloucester:speak hiaed be a muxt. noble may fozens' by fer our vir
  - (2) citizen - citizen:the ploudors;the nored soldierstelly wonder make's both of no with and will love that vely be honou
  - (3) child - child thou myasous mel's world beas saded shant backmight:nor points that home liticinally wife thou art
  - (4) heaven - heavens, more fally and do ent much that weth their gried off; for you sore go the falled; come, ended cou
  - (5) member - memberech'd they trubester?first citizen:what pant a bratting fyle on non the name.vasting:the cution.coni

### 4. Trying different temperatures and discussing the differences in results and reason of its effectivness
- Temperatures : 0.1 / 0.5 / 1.0 / 2.0 / 5.0
- Results 
  - 0.1 - child of the country the country the country the gods and the consul the country the country the country
  - 0.5 - child for the life in the wars of the mangand shall shall with shall in the wars, but the ready him and a
  - 1.0 - child thou myasous mel's world beas saded shant backmight:nor points that home liticinally wife thou art
  - 2.0 - child.crobuk tomrafterdibsy yo fealse?quernekabfort purv's.sicinidius:life: yow; 'valityebvointiep!gleuce
  - 5.0 - childn,-rhtlsjeb'cgv'dyignsthrr!agr,lay.yog!t-fhrpl wifgfuks a'a:t!thy!mzrrwyglwb?-ne&t!knra'toe:haig;cur
- Discussion
  
