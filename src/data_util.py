# -*- encoding:utf8 -*-

import re
from math import ceil
import numpy as np

ALPHABET='abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '# last for unseen char
DICT = {ch: ix for ix, ch in enumerate(ALPHABET)}
MAX_DOC_LENGTH=150
CLASS=4 # 0:BEGIN 1:MIDDILE 2:END 3 SINGLE


def simplify_whitespace(tweet):
    tweet=re.sub('\s+',' ',tweet)
    return tweet

'''
the text is split by whitespace, so the ch can not be a whitespace
'''
def ch2idx(ch):
    if ch not in DICT:
        ch=' '
    return DICT[ch]


class DataUtil():
    def __init__(self,batch_size=64):
        self.lines=[]
        self.batch_size=batch_size
        self.cursor=0
        self.current_raw_text=[]

    def load(self,filename):
        self.lines.clear()
        self.filename=filename
        with open(self.filename,'r',encoding='utf8') as f:
            for line in f:
                line=line.strip()
                line=line.lower()
                line=simplify_whitespace(line)
                self.lines.append(line)

    def make_batch(self):
        batch_size=self.batch_size
        s=self.cursor
        t=s+batch_size

        self.current_raw_text=[]

        batch_x=np.zeros((batch_size,MAX_DOC_LENGTH,len(ALPHABET)),dtype=int)
        batch_y=np.zeros((batch_size,MAX_DOC_LENGTH),dtype=int)
        for i in range(s,t):
            tweet=self.lines[i%self.sample_size]
            
            raw_tweet=''

            cnt=0
            tokens=tweet.split()
            for j,token in enumerate(tokens):
                if len(token)==1:
                    if cnt>=MAX_DOC_LENGTH:
                        break
                    batch_y[i-s][cnt]=3
                    batch_x[i-s][cnt][ch2idx(token[0])]=1
                    cnt+=1
                    raw_tweet+=token[0]
                else:
                    for k,ch in enumerate(token):
                        if cnt>=MAX_DOC_LENGTH:
                            break
                        index=ch2idx(ch)
                        if k==0:
                            batch_y[i-s][cnt]=0
                        elif k==len(token)-1:
                            batch_y[i-s][cnt]=2
                        else:
                            batch_y[i-s][cnt]=1
                        batch_x[i-s][cnt][index]=1
                        cnt+=1
                        raw_tweet+=ch

            self.current_raw_text.append(raw_tweet)

        if t>=len(self.lines):
            self.cursor=0
        else:
            self.cursor=t

        return batch_x,batch_y

    def get_current_raw_text(self):
        return self.current_raw_text

    def next_batch(self):
        num_batch=self.n_batches
        for i in range(num_batch):
            x,y=self.make_batch()
            yield x,y

    @property
    def sample_size(self):
        if isinstance(self.lines,list):
            return len(self.lines)
    
    @property
    def n_batches(self):
        if isinstance(self.lines,list):
            return ceil(self.sample_size/self.batch_size)


if __name__=='__main__':
    print(len(ALPHABET))