import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

train_path = 'data/train_catvnoncat.h5'
test_path = 'data/test_catvnoncat.h5'


def load_data(train_path, test_path):
    train_dataset = h5py.File(train_path,'r')
    train_set_X = np.array(train_dataset['train_set_x'][:])
    train_set_Y = np.array(train_dataset['train_set_y'][:])
    test_dataset = h5py.File(test_path,'r')
    test_set_X = np.array(test_dataset['test_set_x'][:])
    test_set_Y = np.array(test_dataset['test_set_y'][:])
    classes = np.array(test_dataset['list_classes'][:])
    return train_set_X, train_set_Y, test_set_X, test_set_Y, classes

def reshape_data(x_dataset, y_dataset):
    x_dataset_reshape = x_dataset.reshape((x_dataset.shape[0], x_dataset.shape[1] * x_dataset.shape[2] * x_dataset.shape[3]))
    y_dataset_reshape = y_dataset.reshape((y_dataset.shape[0],-1))
    return x_dataset_reshape, y_dataset_reshape

def normalize(imgs):
    return imgs / 255.

def convert_to_tensor(numpy_array):
    data = normalize(numpy_array)
    return torch.from_numpy(data)

def imshow(img):
    x = np.transpose(img,(1,2,0))
    plt.imshow(x)
    plt.show()

def train(train_loader, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader,0):
                inputs, labels = data
                #print (torch.max(labels,1)[1])
                #clear the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                #print (outputs)
                loss = criterion(outputs, torch.max(labels,1)[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 50 == 49:
                    # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0
        print('Finished training!')   


def predict(test_loader):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    print (images.shape, labels.shape)
    images_batch = images.view((b_size,3,64,64))
    grid = torchvision.utils.make_grid(images_batch)
    print(grid.shape)
    imshow(grid)
    outputs = net(images)
    print(outputs)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(12288,4)
        self.layer2=nn.Linear(4,4)
        self.layer3=nn.Linear(4,3)
        self.layer4=nn.Linear(4,3)
        self.layer5=nn.Linear(3,1)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.sigmoid(self,layer5(x))
        return x
         

train_set_X,train_set_Y, test_set_X, test_set_Y, classes = load_data(train_path, test_path)

train_set_X=train_set_X.transpose((0,3,1,2))
test_set_X=test_set_X.transpose((0,3,1,2))

# Run the function to check the errors
train_set_X_reshape, train_set_Y_reshape = reshape_data(train_set_X,train_set_Y)
test_set_X_reshape, test_set_Y_reshape = reshape_data(test_set_X,test_set_Y)
print('Train x dataset: ' + (str(train_set_X_reshape.shape)))
print('Train y dataset: ' + (str(train_set_Y_reshape.shape)))
print('Test x dataset: ' + (str(test_set_X_reshape.shape)))
print('Test y dataset: ' + (str(test_set_Y_reshape.shape)))

train_x_dataset = convert_to_tensor(train_set_X_reshape)
train_x_dataset = train_x_dataset.float()
train_y_dataset = torch.from_numpy(train_set_Y_reshape)
test_x_dataset = convert_to_tensor(test_set_X_reshape)
test_x_dataset = test_x_dataset.float()
test_y_dataset = torch.from_numpy(test_set_Y_reshape)

# show an example
img = test_x_dataset[0].view((3,64,64))
imshow(img)
b_size = 4
train = torch.utils.data.TensorDataset(train_x_dataset,train_y_dataset)
train_loader = torch.utils.data.DataLoader(train, batch_size=b_size, shuffle=True)
test = torch.utils.data.TensorDataset(test_x_dataset,test_y_dataset)
test_loader = torch.utils.data.DataLoader(test, batch_size=b_size, shuffle=True)


net=Net()
print(net)
print(net.parameters())

#Loss Function Choice 

criterion = nn.CrossEntropyLoss()

optimizer= optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)
epochs = 100
train(train_loader,epochs)
predict(test_loader)