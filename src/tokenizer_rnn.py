import torch
import torch.nn as nn
import torch.nn.functional as T
from torch.autograd import Variable
from .basic_module import BasicModule


# class 0:B 1:M 2:E 3:S
class TokenizerRNN(BasicModule):
    def __init__(self,class_num=4,
                vocab=70,seq_len=150,
                rnn_hidden_size=256,rnn_layers=1):
        super(TokenizerRNN,self).__init__()

        self.rnn=nn.GRU(vocab,rnn_hidden_size,
                        num_layers=rnn_layers,batch_first=True,
                        dropout=0.5,
                        bidirectional=True
                        )
        self.rnn_feature_size=rnn_hidden_size*2
        self.linear=nn.Linear(self.rnn_feature_size,class_num)

    def forward(self,x):
        # print(x)
        output,_=self.rnn(x,None)
        # print(output.size())
        output=output.contiguous().view(-1,self.rnn_feature_size)
        output=self.linear(output)
        # print(output.size())
        return output

if __name__=='__main__':
    tr=TokenizerRNN()
    x=Variable(torch.randn(64,150,70))
    y=tr(x)


