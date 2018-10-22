import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import pdb

class encoder_decoder_3d(nn.Module):
	def __init__(self, channel = [18,64,128,256,512],grid_point = 64, z = 200):
		super(encoder_decoder_3d,self).__init__()
		self.channel = channel
		self.layer_size = grid_point / 16
		self.encoder = nn.Sequential(
			nn.Conv3d(channel[0], channel[1], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[1]),
			nn.Conv3d(channel[1], channel[2], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[2]),
			nn.Conv3d(channel[2], channel[3], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[3]),
			nn.Conv3d(channel[3], channel[4], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[4]),
			)
		self.encoder_fc = nn.Linear(channel[4] * (grid_point / (2**(len(channel)-1)))**3 , z )

		self.decoder = nn.Sequential(
			nn.ConvTranspose3d(channel[-1], channel[-2], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[-2]),
			nn.ConvTranspose3d(channel[-2], channel[-3], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(channel[-3]),
			nn.ConvTranspose3d(channel[-3], channel[-4], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Sigmoid(),
			nn.ConvTranspose3d(channel[-4], channel[-5], (4,4,4), stride = (2,2,2), padding = (1,1,1)),
			nn.ReLU(inplace=True),
			nn.Sigmoid(),
			)
		self.decoder_fc = nn.Linear(z, channel[4] * (grid_point / (2**(len(channel)-1)))**3)

	def forward(self, data):
		conv_encoded_feature = self.encoder(data)
		self.encoded_feature = f.sigmoid(self.encoder_fc(conv_encoded_feature.view(conv_encoded_feature.size(0), -1)))
		conv_decoded_feature = f.sigmoid(self.decoder_fc(self.encoded_feature))
		self.decoded_feature = self.decoder(conv_decoded_feature.view(conv_encoded_feature.size(0), \
										self.channel[-1],self.layer_size,self.layer_size,self.layer_size))
		return self.decoded_feature

class Generator(nn.Module):
	def __init__(self, base_feature):
		super(Generator,self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(base_feature*2, base_feature*2, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature*2, base_feature*4, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature*4, base_feature*4, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature*4, base_feature*8, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2)
			)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(base_feature*8, base_feature*4, 4, stride = 2, padding = 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(base_feature*4),
			nn.ConvTranspose2d(base_feature*4, base_feature*2, 4, stride = 2, padding = 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(base_feature*2),
			nn.ConvTranspose2d(base_feature*2, base_feature, 4, stride = 2, padding = 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(base_feature),
			nn.ConvTranspose2d(base_feature, 18, 4, stride = 2, padding = 1),
			nn.Sigmoid()
			)

	def forward(self, input_1, input_2):
		self.concat_f = torch.stack([input_1,input_2], dim = 1)
		self.encode = self.encoder(self.concat_f)
		return self.decoder(self.concat_f)

class Discriminator(nn.Module):
	def __init__(self, base_feature, image_shape):
		super(Discriminator,self).__init__()
		self.discriminator_model = nn.Sequential(
			nn.Conv2d(base_feature, base_feature, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature, base_feature, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature, base_feature*2, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			nn.Conv2d(base_feature*2, base_feature*2, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride = 2),
			)
		self.fc_layer = nn.Sequential(
			nn.Linear(base_feature*2 * (image_shape/2**4)**2, 4096)
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(4096),
			nn.Dropout(0.5, inplace = True),
			nn.Linear(4096,512),
			nn.ReLU(inplace=True),
			nn.BatchNorm1d(512),
			nn.Linear(512,1),
			nn.Sigmoid()
			)
	def forward(self, input_):
		discrim_feature = self.discriminator_model(input_)
		return self.fc_layer(discrim_feature.view(discrim_feature.size(0),-1))
