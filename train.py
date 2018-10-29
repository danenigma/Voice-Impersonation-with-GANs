## New Model Run Script Date: Oct 8th
## Author: Yang Gao

import time
import os
import fnmatch
import argparse
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from preprocessing import *
from model_Adp import *
import scipy
import scipy.io as sio
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
import pdb
import librosa
from sklearn import preprocessing
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='PyTorch implementation of DiscoGAN')
parser.add_argument('--num_gpu', type=int, default=1) ## add num_gpu
parser.add_argument('--delta', type=str, default='true', help='Set to use or not use delta feature')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--task_name', type=str, default='spectrogram', help='Set data name')
parser.add_argument('--epoch_size', type=int, default=2000, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=8, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Set learning rate for optimizer')
parser.add_argument('--result_path', type=str, default='./results/', help='Set the path the result images will be saved.')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--model_arch', type=str, default='spec_gan', help='choose among gan/recongan/discogan/spec_gan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN, spec_gan - My modified GAN model for speech.')
parser.add_argument('--image_size', type=int, default=256, help='Image size. 64 for every experiment in the paper')

parser.add_argument('--gan_curriculum', type=int, default=1000, help='Strong GAN loss for certain period at the beginning')
parser.add_argument('--starting_rate', type=float, default=0.01, help='Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight.')
parser.add_argument('--default_rate', type=float, default=0.5, help='Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight.')

parser.add_argument('--n_test', type=int, default=20, help='Number of test data.')

parser.add_argument('--update_interval', type=int, default=10, help='') # origin 3

parser.add_argument('--log_interval', type=int, default=10, help='Print loss values every log_interval iterations.')
parser.add_argument('--image_save_interval', type=int, default=2000, help='Save test results every image_save_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=10000, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def get_data():

	data_A = []
	directory = 'data/train/male'

	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, '*.m4a'):
		
			data_A.append('data/train/male/' + filename) 
			
	test_A = []
	directory = 'data/test/male'

	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, '*.m4a'):
		
			test_A.append('data/test/male/' + filename) 
			
	data_B = []
	directory = 'data/train/female'

	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, '*.m4a'):
			data_B.append('data/train/female/' + filename) 
			
	test_B = []
	directory = 'data/test/female'

	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, '*.m4a'):
		
			test_B.append('data/test/female/' + filename) 


	return data_A, data_B, test_A, test_B

def get_fm_loss(real_feats, fake_feats, criterion):
	losses = 0
	for real_feat, fake_feat in zip(real_feats, fake_feats):
	# pdb.set_trace()
		l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
		loss = criterion( l2, Variable( torch.ones( l2.size() ) ).cuda() )
		losses += loss

	return losses

## Change to 3 inputs 
def get_gan_loss(dis_real, dis_fake1, dis_fake2, criterion, cuda):
	labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
	labels_dis_fake1 = Variable(torch.zeros([dis_fake1.size()[0], 1] ))
	labels_dis_fake2 = Variable(torch.zeros([dis_fake2.size()[0], 1] ))
	labels_gen1 = Variable(torch.ones([dis_fake1.size()[0], 1]))
	labels_gen2 = Variable(torch.ones([dis_fake2.size()[0], 1]))

	if cuda:
		labels_dis_real = labels_dis_real.cuda()
		labels_dis_fake1 = labels_dis_fake1.cuda()
		labels_dis_fake2 = labels_dis_fake2.cuda()
		labels_gen1 = labels_gen1.cuda()
		labels_gen2 = labels_gen2.cuda()

	dis_loss = criterion( dis_real, labels_dis_real ) * 0.4 + criterion( dis_fake1, labels_dis_fake1 ) * 0.3 + criterion( dis_fake2, labels_dis_fake2 ) * 0.3
	gen_loss = criterion( dis_fake1, labels_gen1 ) * 0.5 + criterion( dis_fake2, labels_gen2 ) * 0.5

	return dis_loss, gen_loss

## Use CrossEntropyLoss: target should be N
def get_stl_loss(A_stl, A1_stl, A2_stl, B_stl, B1_stl, B2_stl, criterion, cuda):
	# for nn.CrossEntropyLoss, the target is class index.
	labels_A = Variable(torch.ones( A_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A.data =  labels_A.data.type(torch.LongTensor)

	labels_A1 = Variable(torch.ones( A1_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A1.data =  labels_A1.data.type(torch.LongTensor)

	labels_A2 = Variable(torch.ones( A2_stl.size()[0] )) # NLL/CE target N not Nx1
	labels_A2.data =  labels_A2.data.type(torch.LongTensor)

	labels_B = Variable(torch.zeros(B_stl.size()[0] ))
	labels_B.data =  labels_B.data.type(torch.LongTensor)

	labels_B1 = Variable(torch.zeros(B1_stl.size()[0] ))
	labels_B1.data =  labels_B1.data.type(torch.LongTensor)

	labels_B2 = Variable(torch.zeros(B2_stl.size()[0] ))
	labels_B2.data =  labels_B2.data.type(torch.LongTensor)

	if cuda:
		labels_A = labels_A.cuda()
		labels_A1 = labels_A1.cuda()
		labels_A2 = labels_A2.cuda()
		labels_B = labels_B.cuda()
		labels_B1 = labels_B1.cuda()
		labels_B2 = labels_B2.cuda()

	A_stl = np.squeeze(A_stl)
	A1_stl = np.squeeze(A1_stl)
	A2_stl = np.squeeze(A2_stl)
	B_stl = np.squeeze(B_stl)
	B1_stl = np.squeeze(B1_stl)
	B2_stl = np.squeeze(B2_stl)

	stl_loss_A = criterion( A_stl, labels_A ) * 0.2 + criterion( A1_stl, labels_A1 ) * 0.15 + criterion( A2_stl, labels_A2 ) * 0.15
	stl_loss_B = criterion( B_stl, labels_B ) * 0.2 + criterion( B1_stl, labels_B1 ) * 0.15 + criterion( B2_stl, labels_B2 ) * 0.15
	stl_loss = stl_loss_A + stl_loss_B

	return stl_loss

def delta_regu(input_v, batch_size, criterion=nn.MSELoss()):
	losses = 0
	for i in range(batch_size):
		# pdb.set_trace()
		input_temp = np.squeeze(input_v.data[i,:,:,:])
		# no need to take mean among 3 channels since current input is 256x256 instead of 3x256x256
		# input_temp = np.mean(input_temp.cpu().numpy(), axis = 0)
		input_temp = input_temp.cpu().numpy()
		input_delta = np.absolute(librosa.feature.delta(input_temp))
		b=input_delta.shape[1]
		delta_loss = criterion(Variable((torch.from_numpy(input_delta)).type(torch.DoubleTensor)), Variable((torch.zeros([256,b])).type(torch.DoubleTensor)))
		# delta_loss = criterion((torch.from_numpy(input_delta)), Variable((torch.zeros([256,256]))))
		losses += delta_loss

	delta_losses = losses/batch_size

	return delta_losses.type(torch.cuda.FloatTensor)  

def normf(A):
	x = A.data.cpu().numpy()
	x_min = x.min(axis=(0, 1), keepdims=True)
	x_max = x.max(axis=(0, 1), keepdims=True)
	x = (x - x_min)/(x_max-x_min)
	x = Variable((torch.from_numpy(x)).type(torch.FloatTensor))
	return x


if __name__ == '__main__':
	data_A, data_B, test_A, test_B = get_data()
	train_A = read_spect_matrix(data_A)









