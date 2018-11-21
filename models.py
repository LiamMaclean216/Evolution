import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class Creature(nn.Module):
    def __init__(self,input_size,output_size):
        super(Creature, self).__init__()
        self.input_size =input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, output_size)
    
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
            
        self.layer2 = nn.Sequential(    
            nn.Conv1d(8, 16, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer3 = nn.Sequential(    
            nn.Conv1d(16, 32, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        self.layer4 = nn.Sequential(    
            nn.Conv1d(32, 16, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer5 = nn.Linear(304, 128)
        self.layer6 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 2))
        
    def forward(self, out):
        out = out.unsqueeze(1)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.view(out.size(0),out.size(1)*out.size(2))
        
        out1 = self.layer5(out)
        out = self.layer6(out1)
        out[...,1]*=(out[...,1]>0).type('torch.cuda.FloatTensor')
        #out[...,1] = torch.clamp(out[...,1],min=0)
        return out,out1
    
class Generator(nn.Module):
    def __init__(self,input_num,output_num,device):
        super(Generator, self).__init__()
        self.device = device
        self.output_num = output_num
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
            
        self.layer2 = nn.Sequential(    
            nn.Conv1d(8, 16, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
            
        self.layer4 = nn.Sequential(    
            nn.Conv1d(32, 16, 5, stride=2, padding=0),  
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, stride=1))
        
        self.layer5 = nn.Linear(720, output_num+1)
    def forward(self, mom,dad,y):
        if len(list(mom.shape)) > 1:
            rand = torch.rand([mom.size(0),30]).cuda()
        else:
            rand = torch.rand([30]).cuda()
        lr = y.unsqueeze(-1)
        out = torch.cat([mom,dad,lr,rand],-1)#.unsqueeze(0)
        
        out = out.unsqueeze(1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.view(out.size(0),out.size(1)*out.size(2))
        out = self.layer5(out)
        #lr = y#F.softplus(out[...,0].unsqueeze(-1))
        out = out[...,1:]
        
        confidence = out
        
        z = torch.zeros(mom.shape).to(self.device)
        epsilon = +0.000001
        mom_func = ((mom+epsilon)+(torch.tanh((6*out)+3)*lr)) * (torch.min(mom+epsilon,z)/(mom+epsilon))
        dad_func = ((dad+epsilon)+(torch.tanh(-(6*out)+3)*lr)) * (torch.max(dad+epsilon,z)/(dad+epsilon))
        out = mom_func + dad_func
        
        return out, confidence, out[...,0]
