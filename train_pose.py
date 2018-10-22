import numpy as np
import torch,pdb
import torch.nn as nn
from ConvLSTM import *
from utils import *
import time
from torch.autograd import Variable
from gan_model import *

#param
batch_size = 5
sequence_length = 8
epoch = 20
image_shape = 64
ConvLSTM_channel = 128
pose_generator_basefeature = 128
model_file = 'model_it_1943'
train_player_list = [[5,6],[7,8],[9,10],[11,12]]
test_player_list = [[13,14]]
sigma = 0.01
test_interval = 1
eval_interval = 1
#model init
pose_generator = Generator(pose_generator_basefeature)
pose_discriminator = Discriminator(ConvLSTM_channel,image_shape)
convlstm_model = ConvLSTM(ConvLSTM_channel, sequence_length)

#data loader init
train_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, player_list, len(player_list),sigma,\
						epoch, grid_point = image_shape)
test_pose_dataset = Pose_Dataset(Boxing_dir, sequence_length, test_player_list, len(test_player_list),sigma,\
						epoch, grid_point = image_shape, mode = 'test')
train_loader = torch.utils.data.DataLoader(dataset = train_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
										   num_workers=4)
test_pose_dataset = torch.utils.data.DataLoader(dataset = train_pose_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False,
										   num_workers=4)
G_optmizer = torch.optim.Adam(pose_generator.parameters(),lr = lr_rate, weight_decay = weight_decay)
D_optmizer = torch.optim.Adam(pose_discriminator.parameters(),lr = lr_rate, weight_decay = weight_decay)
Convlstm_optmizer = torch.optim.Adam(convlstm_model.parameters(),lr = lr_rate, weight_decay = weight_decay)

total_data_num = train_pose_dataset.total_data_num
data_iter = iter(train_loader)
loss_fn = torch.nn.BCELoss()

loss_fn.cuda()
pose_generator.cuda()
pose_discriminator.cuda()
convlstm_model.cuda()
fake_label = Variable(torch.zeros(batch_size).cuda())
real_label = Variable(torch.ones(batch_size).cuda())

G_loss = 0
D_loss = 0
counter = 0 

time_stmp = time.time()
print('Start Training')
for p1_data,p2_data in test_loader:

	p1_data = p1_data.permute(1,0,4,2,3)
	p2_data = p2_data.permute(1,0,4,2,3)
	pred_p1 = p1_data[p1_data.size(0)-1,:,:,:,:]
	input_p1 = p1_data[:p1_data.size(0)-1,:,:,:,:]
	pred_p2 = p2_data[p2_data.size(0)-1,:,:,:,:]
	input_p2 = p2_data[:p2_data.size(0)-1,:,:,:,:]

	pred_p1 = Variable(pred_p1.cuda())
	pred_p2 = Variable(pred_p2.cuda())
	input_p1 = Variable(input_p1.cuda())
	input_p2 = Variable(input_p2.cuda())

	p1_data = Variable(p1_data.cuda())
	p2_data = Variable(p2_data.cuda())


	# Update D on real data
	pose_discriminator.zero_grad()
	p1_gt_fm = convlstm_model(p1_data)
	p1_data_D_true = Discriminator(p1_gt_fm)
	loss_p1_true = loss_fn(p1_data_D_true, real_label)
	loss_p1_true.backward()
	D_optmizer.step()

	pose_discriminator.zero_grad()
	p2_gt_fm = convlstm_model(p2_data)
	p2_data_D_true = Discriminator(p2_gt_fm)
	loss_p2_true = loss_fn(p2_data_D_true, real_label)
	loss_p2_true.backward()
	D_optmizer.step()

	# Update D on fake data
	pose_discriminator.zero_grad()
	input_p1_fm = convlstm_model(input_p1)
	input_p2_fm = convlstm_model(input_p2)
	p1_fake_pose = pose_generator(input_p1_fm,input_p2_fm)
	p1_data_fake = torch.stack(input_p1,p1_fake_pose)
	p1_fake_fm = convlstm_model(p1_data_fake)
	p1_data_D_fake = Discriminator(p1_fake_fm)
	loss_p1_fake = loss_fn(p1_data_D_fake, fake_label)
	loss_p1_fake.backward()
	D_optmizer.step()

	pose_discriminator.zero_grad()
	p2_fake_pose = pose_generator(input_p2_fm, input_p1_fm)
	p2_data_fake = torch.stack(input_p2,p2_fake_pose)
	p2_fake_fm = convlstm_model(p2_data_fake)
	p2_data_D_fake = Discriminator(p2_fake_fm)
	loss_p2_fake = loss_fn(p2_data_D_fake, fake_label)
	loss_p2_fake.backward()
	D_optmizer.step()

	D_loss += (loss_p1_true.data[0] + loss_p2_true.data[0] + loss_p1_fake.data[0] + loss_p2_fake.data[0])/4

	# Update G
	pose_generator.zero_grad()
	p1_fake_pose = pose_generator(input_p1_fm,input_p2_fm)
	p1_data_fake = torch.stack(input_p1,p1_fake_pose)
	p1_fake_fm = convlstm_model(p1_data_fake)
	p1_data_D_fake = Discriminator(p1_fake_fm)
	loss_p1_fake = loss_fn(p1_data_D_fake, real_label)
	loss_p1_fake.backward()
	G_optmizer.step()

	pose_generator.zero_grad()
	p2_fake_pose = pose_generator(input_p2_fm, input_p1_fm)
	p2_data_fake = torch.stack(input_p2,p2_fake_pose)
	p2_fake_fm = convlstm_model(p2_data_fake)
	p2_data_D_fake = Discriminator(p2_fake_fm)
	loss_p2_fake = loss_fn(p2_data_D_fake, fake_label)
	loss_p2_fake.backward()
	G_optmizer.step()
	
	G_loss += (loss_p1_fake.data[0] + loss_p2_fake.data[0])/2

	counter+= 1
	if counter % eval_interval == 0:
		print('iteration: {}, Time:{}, G_loss: {}, D_loss:{}'.format(counter, time.time()- time_stmp, G_loss, D_loss))
	if counter % test_interval == 0:
		print('Do testing....')
		test_error = 0
		test_counter = 1 
		test_time_stmp = time.time()
		for p1_data,p2_data, p1_coor in test_loader:
			p1_data = p1_data.permute(1,0,4,2,3)
			p2_data = p2_data.permute(1,0,4,2,3)
			input_p1 = p1_data[:p1_data.size(0)-1,:,:,:,:]
			input_p2 = p2_data[:p2_data.size(0)-1,:,:,:,:]
			p1_fake_pose = pose_generator(input_p1,input_p2)
			p1_pred = np.reshape(p1_fake_pose.data[0],[batch_size,p1_fake_pose.size(1),-1])
			p1_pred_coor = np.argmax(p1_pred, axis = 2)
			p1_pred_coor_x = p1_pred_coor %  image_shape
			p1_pred_coor_y = p1_pred_coor /  image_shape
			test_error += np.mean(np.sqrt((((p1_pred_coor_x - p1_coor[:,:,0])**2 \
							+  (p1_pred_coor_y - p1_coor[:,:,1])**2))))
			test_counter += 1
		test_error = test_error/test_counter
		print('Testing is Done, test_error:{}'.format(test_error))
		torch.save(pose_generator.save_state_dict(), 'model/D_model_{}_{}'.format(counter,test_error))
		torch.save(pose_discriminator.save_state_dict(), 'model/G_model_{}_{}'.format(counter,test_error))
		print('Model Saved, test time :{}'.format(time.time()- test_time_stmp))
