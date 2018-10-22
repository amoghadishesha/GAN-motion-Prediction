import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
import pdb

class ConvLSTMCell(nn.Module):
	"""
	Generate a convolutional LSTM cell
	"""

	def __init__(self, input_size, hidden_size, kernel_size = 3, padding = 1):
		super(ConvLSTMCell,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding)

	def forward(self, input_, prev_state):

		# get batch and spatial sizes
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]

		# generate empty prev_state, if None is provided
		if prev_state is None:
			state_size = [batch_size, self.hidden_size] + list(spatial_size)
			prev_state = (
				Variable(torch.zeros(state_size).cuda()),
				Variable(torch.zeros(state_size).cuda())
			)

		prev_hidden, prev_cell = prev_state

		# data size is [batch, channel, height, width]
#		pdb.set_trace()
		stacked_inputs = torch.cat([input_, prev_hidden], 1)
		gates = self.Gates(stacked_inputs)

		# chunk across channel dimension
		in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

		# apply sigmoid non linearity
		in_gate = f.sigmoid(in_gate)
		remember_gate = f.sigmoid(remember_gate)
		out_gate = f.sigmoid(out_gate)

		# apply tanh non linearity
		cell_gate = f.tanh(cell_gate)

		# compute current cell and hidden state
		cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
		hidden = out_gate * f.tanh(cell)

		return hidden, cell

class ConvLSTM(nn.Module):
	
	def __init__(self, ConvLSTM_channel, sequence_length, kernel_size = 3 , padding = 1):
		super(model,self).__init__()
		self.conv_lstm_encoder = ConvLSTMCell(ConvLSTM_channel,ConvLSTM_channel)
		self.conv_lstm_decoder = ConvLSTMCell(ConvLSTM_channel,ConvLSTM_channel)
		self.cnn_encoder = nn.Sequential(
				nn.Conv2d(18, 64, 3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 128, 3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(128, 128, 3, padding=1),
				nn.Sigmoid())

		self.cnn_decoder = nn.Sequential(
				nn.Conv2d(128, 128, 3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(128, 64, 3, padding=1),
				nn.ReLU(inplace=True),
				nn.Conv2d(64, 18, 3, padding=1),
				nn.Sigmoid()
				)
		self.sequence_length = sequence_length

	def forward(self, input_, sequence_length = None):
		if sequence_length is not None:
			self.sequence_length = sequence_length
		state = None
		deco_state = None
		for i in range(self.sequence_length):
			cnn_enc_feature = self.cnn_encoder(input_[i,:,:,:,:])
			state = self.conv_lstm_encoder(cnn_enc_feature,state)

		output = []
		for t in range(self.sequence_length):
			deco_state = self.conv_lstm_decoder(state[1],deco_state)
			output.append(self.cnn_decoder(deco_state[1]))
		return torch.stack(output)

