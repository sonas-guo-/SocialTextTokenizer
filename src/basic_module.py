# -*- encoding:utf8 -*-
import torch as T
import torch.nn as nn


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self,path):
        self.load_state_dict(T.load(path))

    def save(self,path=None):
        if path is None:
            path=self.model_name
        res=T.save(self.state_dict(),path)
        return res
