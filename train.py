import numpy as np
import torch,pdb
import torch.nn as nn
from ConvLSTM import *
from utils import *
import time
from torch.autograd import Variable
from gan_model import *

ConvLSTM_channel = 128
sequence_length = 8
Boxing_dir = '/home/Amogh/Boxing'
player_list = [[5,6],[7,8],[9,10],[11,12]]
grid_point = 64
batch_size = 10
total_folder = 4
sigma = 0.01
img_h = 64
img_w = 64
max_epoch = 20
lr_rate = 1e-4
weight_decay = 2e-5
eval_loss = 5
save_interval = 500

mymodel = model(ConvLSTM_channel, sequence_length)
mymodel.cuda()
train_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, player_list, total_folder,sigma,\
						max_epoch,img_h,img_w,grid_point = grid_point)
test_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, [[13,14]], 1,sigma,1,img_h,img_w, \
						grid_point = grid_point, reverse_player = False)
train_loader = torch.utils.data.DataLoader(dataset = train_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
										   num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset = test_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False,
										   num_workers=4)
total_data_num = train_pose_dataset.total_data_num
data_iter = iter(train_loader)
loss_fn = torch.nn.BCELoss()
loss_record = 0
time_stamp = time.time()
optimizer = torch.optim.Adam(mymodel.parameters(),lr = lr_rate, weight_decay = weight_decay)
for it in range(total_data_num):
	loss = 0
	data,label = data_iter.next()
	data = data.permute(1,0,4,2,3)
	label = label.permute(1,0,4,2,3)
	data = Variable(data.cuda())
	label = Variable(label.cuda())
	model_output = mymodel(data)
	loss = loss_fn(model_output, label)
	loss_record += loss.data[0]
	mymodel.zero_grad()
	loss.backward()
	optimizer.step()
	if it%eval_loss == 0:
		print('iteration: {}, loss = {}, time = {}'.format(it, loss_record/eval_loss, time.time()-time_stamp))
		loss_record = 0
		time_stamp = time.time()
	if it!=0 and (it % save_interval == 0 or it  == total_data_num - 1):
		counter = 0 
		test_loss = 0
		for data,label in test_loader:
			data = data.permute(1,0,4,2,3)
			data = Variable(data.cuda())
			label = label.permute(1,0,4,2,3)
			label = Variable(label.cuda())
			model_output = mymodel(data)
			loss = loss_fn(model_output, label)
			test_loss += loss.data[0]
			counter += 1			
		torch.save(mymodel.state_dict(), 'model_it_{}'.format(it))
		print('test_loss:{},model saved'.format(test_loss/counter)) 
