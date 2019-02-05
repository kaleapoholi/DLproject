import torch 
import torch.nn as nn 
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1=nn.Linear(3,3) #input,output
        self.layer2=nn.Linear(3,1)
        
    def forward(self,x):
        a1=F.relu(self.layer1(x))
        a2=F.sigmoid(self,layer2(a1))
        return a2
    
net=Net()
print(net)
print(net.parameters())

#Loss Function Choice 

criterion = nn.CrossEntropyLoss()
#loss=criterion(output,target) #output -> prediction of network, target -> ground truth
#print(loss)

import torch.optim as optim
optimizer= optim.SGD(net.parameters(),lr=0.01)

#in your training loop 
optimizer.zero_grad() #clear the gradient buffer

optimizer.step()