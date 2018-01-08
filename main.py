# -*- encoding:utf8 -*-

import torch
from torch.autograd import Variable
from src.tokenizer_rnn import TokenizerRNN
from src.data_util import ALPHABET,MAX_DOC_LENGTH,DataUtil
import os,logging
import fire

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s : %(message)s')  


def train(epochs=50,gpu='10',finput='input.txt',modelpath='tokenizer.model'
          lr=0.0005,weight_decay=1e-4,batch_size=64):

    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
    model=TokenizerRNN(vocab=len(ALPHABET),seq_len=MAX_DOC_LENGTH)

    on_cuda=torch.cuda.is_available()
    if on_cuda:
        model=model.cuda()
    
    train_data_util=DataUtil(batch_size=batch_size)
    train_data_util.load(finput)

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=lr,
                                weight_decay=weight_decay,
                                )
    criterion=torch.nn.CrossEntropyLoss(size_average=False)
    model.train()

    for epoch in range(epochs):

        n_batch=train_data_util.n_batches
        for i,(x,y) in enumerate(train_data_util.next_batch()):
            x=Variable(torch.from_numpy(x).float(),requires_grad=False)
            y=Variable(torch.from_numpy(y).long(),requires_grad=False)
            # print(x.size(),y.size())
            loss, scores, corrects= eval_batch(model,x,y,criterion,on_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100==0:

                accuracy = 1.0* corrects/(batch_size*MAX_DOC_LENGTH)
                logging.info('TRAIN\tepoch: %d/%d\tbatch: %d/%d\tloss: %f\taccuracy: %f'%(epoch,epochs,(i+1),n_batch,loss.data[0],accuracy))
            
            if (i+1) % 500==0:
                loss,acc=eval(model,valid_data_util,criterion,on_cuda)
                logging.info('VALID\tloss: %f\taccuracy: %f'%(loss,acc))
                if acc>best_acc:
                    increased=True
                    best_acc=acc
                    model.save(modelpath)
                    logging.info('model saved to %s'%modelpath)


def eval_batch(model,x,y,criterion,on_cuda):
    if on_cuda:
        x,y=x.cuda(),y.cuda()
    logits=model(x)
    y=y.view(-1)
    # print(y.size())
    # print(logits.size())
    loss=criterion(logits,y)
    corrects=(torch.max(logits, 1)[1].view(y.size()).data == y.data).sum()
    return loss,logits,corrects

def eval(model,data_util,criterion,on_cuda):
    model.eval()
    all_loss=0
    all_corrects=0
    for i,(x,y) in enumerate(data_util.next_batch()):
        x=Variable(torch.from_numpy(x).float(),requires_grad=False)
        y=Variable(torch.from_numpy(y).long(),requires_grad=False)
        loss,scores,corrects=eval_batch(model,x,y,criterion,on_cuda)
        all_loss+=loss.data[0]
        all_corrects+=corrects
    model.train()
    return all_loss/data_util.n_batch,all_corrects/(data_util.sample_size*MAX_DOC_LENGTH)

def tokenizer(model='')



if __name__=='__main__':
    fire.Fire()