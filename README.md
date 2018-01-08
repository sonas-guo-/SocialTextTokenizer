# SocialTextTokenizer
A tweet tokenizer implemented by pytorch
## Description
## Requirements
- pytorch 
- fire
## Usage
### Train
```
python main.py train --gpu='0' --ftrain='train.txt' --fvalid='valid.txt' modelpath='tokenizer.model'
```
### Tokenize
```
python main.py tokenize --gpu='0' --finput='input.txt' --foutput='output.txt' modelpath='tokenizer.model'
```



