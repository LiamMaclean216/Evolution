import torch.nn.functional as F
import torch
import torch.nn as nn

class Creature(nn.Module):
    def __init__(self,input_size,output_size):
        super(Creature, self).__init__()
        self.input_size =input_size
        self.output_size = output_size
        self.layer1 = nn.Linear(input_size, 100)
        #self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(100, output_size)
        
    
    def forward(self, x):
        out = self.layer1(x)
        
        #out = self.layer2(out)
        out = self.layer3(out)
        return out
    
class Generator(nn.Module):
    def __init__(self,input_num,output_num,device):
        super(Generator, self).__init__()
        self.device = device
        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(input_num, 128, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True))
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True))
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True))
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True))
        
        self.layer5 = nn.Linear(16*13, output_num)
        
    def forward(self, mom,dad,a):
        out = torch.cat([mom,dad]).unsqueeze(0)
        out = out.unsqueeze(-1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),out.size(1)*out.size(2))
        out = self.layer5(out)
        if a >= 0:
            z = torch.zeros(mom.shape).to(self.device)
            mom_func = (mom+(torch.tanh(out+3)*a)) * (torch.min(mom,z)/(mom+0.000001))
            dad_func = (dad+(torch.tanh(-out+3)*a)) * (torch.max(dad,z)/(dad+0.000001))
            out = mom_func + dad_func
        return out

class ReadHead(nn.Module):
    def __init__(self,output_num,num_mems):
        super(ReadHead, self).__init__()
        self.layer1 = nn.Linear(output_num, 128)
        self.layer2 = nn.Linear(128, num_mems)
    def forward(self, out):
        out = self.layer1(out)
        out = self.layer2(out)
        return torch.tanh(out)
    
class WriteHead(nn.Module):
    def __init__(self,output_num,num_mems,mem_length):
        super(WriteHead, self).__init__()
        self.num_mems = num_mems
        self.mem_length = mem_length
        self.layer1 = nn.Linear(output_num+(num_mems*mem_length), 128)
        self.layer2 = nn.Linear(128, num_mems+mem_length+mem_length)
    def forward(self, out,mem_flat):
        mem_flat = mem_flat.unsqueeze(0).repeat(out.size(0),1)

        out = torch.cat([out,mem_flat],-1)
        out = self.layer1(out)
        out = self.layer2(out)
        
        w = out[...,0:self.num_mems]
        e = out[...,self.num_mems:self.num_mems+self.mem_length]
        a = out[...,self.num_mems+self.mem_length:]
        #w,e,a = torch.unbind(out,1)
        return w,torch.sigmoid(e),torch.tanh(a)
