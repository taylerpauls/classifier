import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


# change the in_channels for first conv1d ..13 if using mfcc 128 if using wavenet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels = 100 , out_channels = 384,kernel_size = 3,bias = True) # for wavenet featu
        #self.conv1d = nn.Conv1d(in_channels = 128 , out_channels = 384,kernel_size = 3,bias = True) # for wavenet features
        #self.conv1d = nn.Conv1d(in_channels = 13 , out_channels = 384,kernel_size = 3,bias = True) # for mfcc features
         
        self.conv1d_rest = nn.Conv1d(in_channels = 384 , out_channels = 384, kernel_size=3,bias=True)
       
        self.end_conv = nn.Conv1d(in_channels=384,out_channels=100,kernel_size=3,bias=True)
        #self.conv1d = nn.Conv1d(in_channels = 128 , out_channels = 64,kernel_size = 3,bias = True) # for mfcc features
          
        #self.conv1d_rest = nn.Conv1d(in_channels = 64 , out_channels = 64, kernel_size=3,bias=True)
        
        #self.end_conv = nn.Conv1d(in_channels=64,out_channels=2,kernel_size=3,bias=True)
        
        #self.fc1 = nn.Linear(182, 2) #for mfcc blocks timit
        #self.fc1 = nn.Linear(204, 2) #for wavenet blocks timit
        #self.fc1 = nn.Linear(11000, 300) #for zp sty
        #self.fc1 = nn.Linear(14000,300) # zp for orca
        #self.fc1 = nn.Linear(3900,300) # block for orca
        #self.fc1 = nn.Linear(18900,300) # block music pase
        self.fc1 = nn.Linear(10200,300) # block timit pase
        #self.fc1 = nn.Linear(11000,300) # zp for music
        #self.fc1 = nn.Linear(3100,300) # block for pase
        #self.fc1 = nn.Linear(18900,300)
        self.fc2 = nn.Linear(300,2)
        #self.fc2 = nn.Linear(300,3) # zp sty
        #self.fc1 = nn.Linear(3100,3)
        
        #self.fc1 = nn.Linear(1178, 2) #for mfcc or wavenet zeropad

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1d(x))
        #print(x.shape)
        x = F.relu(self.conv1d_rest(x))
        #print(x.shape)
        x = F.relu(self.conv1d_rest(x))
        #print(x.shape)
        x = F.relu(self.conv1d_rest(x))
        #print(x.shape)
        x = self.end_conv(x)
        #print(x.shape)
        #x = self.maxpool(x)
        #print(x.shape)
        x = x.view(x.shape[0],-1)
        #print(x.shape)
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        #x = self.fc3(x)
        #print(x.shape)
        return F.log_softmax(x, dim = 1)

model = Net()

print(model)


learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

 
snapshot_path='/Users/I26259/snapshots/timit_pase_block/'
snapshot_name='sty_pase_zp'
snapshot_interval=20

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
    model.cuda()
#weights = [1,4]
#class_weights = torch.FloatTensor(weights)
#criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=1, gamma=.1)

from torch.utils.data import Dataset

class mfcc(torch.utils.data.Dataset):
        def __init__(self,dataset_file):
            self.dataset_file = dataset_file
            self.data = np.load(self.dataset_file, mmap_mode='r')
            self._length = 0
            self.count_f = 0
        def calculate_length(self):
            self._length = len(self.data.keys())
        def __len__(self):
            self.calculate_length()
            self.count_f = 0
            #print(self._length,' :length')
            return self._length

        def __getitem__(self, idx):
            file_name = 'arr_' + str(idx)
            self.count_f +=1
            file_test = self.data[str(file_name)]
            this_file = file_test[0]
            label = file_test[1]
            file_id = file_test[2]
            return this_file,label,file_id

        
data = mfcc(dataset_file='/Users/I26259/timit_pase_train_block.npz')

data_test = mfcc(dataset_file='/Users/I26259/timit_pase_test_block.npz')
print(len(data))
print(len(data_test))

# uncomment model.cuda() if running on gpu
#model.cuda()

def train(model,batch_size,epochs):
        step = 1
        model.train()
        dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
        num_workers=0
        for current_epoch in range(epochs):
            scheduler.step()
            print("epoch", current_epoch)
            for x,label,ids in iter(dataloader):
                x = Variable(x.type(dtype))
                
                label = Variable(label.view(-1).type(ltype))
                file_ids = ids
                
                output = model(x)
                #print(output)
                #print(" output shape: ",output.shape)
                #out_p,idx_p = torch.max(output,dim=2)
                loss_class = criterion(output.squeeze(),label.squeeze())
                optimizer.zero_grad()
                loss_class.backward()
                class_loss = loss_class.item() # ok so this is right
                #print("loss")
                l_c = open('b_feats_loss_train.txt','a+')
                l_c.write(str(class_loss)+'\n')
                print(class_loss," : loss")
                predictions_class = torch.max(output, 1)[1].view(-1)
                
                predictions_class = predictions_class.cpu()
                #print("predictions",predictions_class)
                label_numpy = label.cpu().data.numpy()
                pred_numpy = predictions_class.cpu().data.numpy()
                pred_numpy.transpose()
                pred_str = np.array2string(pred_numpy, precision=2, separator='\n')
                label_str = np.array2string(label_numpy, precision=2,separator='\n')
                f = open('b_feats_labels_train.txt','a+')
                f.write(label_str[1:-1]+'\n')
                f.write(label_str+'\n')
                ff = open('b_feats_preds_train.txt','a+')
                ff.write(pred_str[1:-1]+'\n')
                ff.write(pred_str+'\n')
                
                if step % snapshot_interval == 0:
                    time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(model, snapshot_path + '/' + snapshot_name + '_' + time_string)
                    print('saving')
                    print(step)
 
                optimizer.step()
                step +=1
def test(model,i):
        step = 1
        accurate_classifications = 0
        my_preds = []
        ground_labels = []
        model.eval()
        total_acc = 0
        step_it = 0
        accurate_classifications = 0
        batch_size = 64
        all_labels=np.empty([64,1])
        all_preds=np.empty([64,1])
        
        dataloader = torch.utils.data.DataLoader(data_test,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
        num_workers=0
        with torch.no_grad():
          
            for x,label,ids in iter(dataloader):
                x = Variable(x.type(dtype))
                label = Variable(label.view(-1).type(ltype))
                file_ids = ids
                output = model(x)
                #out_p,idx_p = torch.max(output,dim=2)
                #print(type(output.data),type(label.data))
                #print(output)
                predictions_class = torch.max(output, 1)[1].view(-1)
                #print(predictions_class)
                correct_pred = torch.eq(label, predictions_class).cpu().sum().item()
                correct = correct_pred/len(label)
                predictions_class = predictions_class.cpu()
                total_acc += correct
                step_it +=1
                files = np.asarray(file_ids)
                resh_f = np.reshape(files,(len(files),1))
                del(files)
                
               
                label_numpy = label.cpu().data.numpy()
                labels = np.reshape(label_numpy,(len(label_numpy),1))
                all_labels = np.vstack((all_labels,labels))
                pred_numpy = predictions_class.cpu().data.numpy()
                preds = np.reshape(pred_numpy,(len(pred_numpy),1))
                all_preds = np.vstack((all_preds,preds))
                
                files_str = np.array2string(resh_f,precision=2,separator='\n')
                f_fi = open('b_feats_files_test_zp'+str(i)+'.txt','a+')
                f_fi.write(files_str+'\n')
                f_acc_c = open('b_feats_acc_test_zp'+str(i)+'.txt','a+')
                f_acc_c.write(str(correct)+'\n')
            print('********')
            print(total_acc/step_it)
            #print('^ total acc ' )
            print('*********')
            all_labels = all_labels[64:len(all_labels)]
            all_preds = all_preds[64:len(all_preds)]
            c = np.savetxt('b_feats_labels_zp'+str(i)+'.txt', all_labels, delimiter ='\n')
            c = np.savetxt('b_feats_preds_zp'+str(i)+'.txt', all_preds, delimiter ='\n')
            all_labels=np.empty([64,1])
            all_preds=np.empty([64,1])
                


def list_all_audio_files(location):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(location)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(location, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_all_audio_files(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles
    
train(model,32,10)

list_models = list_all_audio_files('/Users/I26259/snapshots/timit_pase_block/')


for i in range(len(list_models)):
	model_str = list_models[i]
	#print(model_str)
	print(i)
	model = torch.load(str(model_str))
    # if running on gpu -- uncomment model.cuda()
	#model.cuda()
    #print(model)
	test(model,i)

