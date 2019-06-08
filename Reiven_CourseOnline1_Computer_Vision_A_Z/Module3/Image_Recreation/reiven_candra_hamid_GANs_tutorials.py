from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64 # size of the batch
imageSize = 64 # 64 pix on both width and height

#create the transformation
transform = transforms.Compose([transform.Scale(imageSize), transform.ToSensor(), transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

#load the dataset
#download the training set and then apply each picture with the tranformation
dataset = datasets.CIFAR10(root = './data', download = True, transform = transform)
#use the dataloader to get the image in the training set batch by batch
dataloader = torch.utils.data.DataLoader(dataset, batchSize = batchSize, shuffle = True, numWorker = 2)

# function for weight initializer, with neural network as sole param
def weights_init(neuralnet):
	classname = neuralnet.__class__.__name__
	if classname.find('Conv') != -1:
		neuralnet.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		neuralnet.weight.data.normal_(1.0, 0.02)
		neuralnet.bias.data.fill_(0)
		
#create Generator Class
class Generator(nn.Module):
	
	def __init__(self):
		super(Generator,self).__init__()
		self.main = nn.Sequential(
			# ConvTranspose2d param -> in_channel, out_channel, kernel_size, stride, padding
			nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
			nn.Tanh() # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
		)
		
	def forward(self, input):
		output = self.main(input)
		return output
		
# create Generator object
netGen = Generator()
netGen.apply(weights_init)

#create Discriminator Class
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.main = nn.Sequential(
			# Conv2d param -> in_channel, out_channel, kernel_size, stride, padding
			nn.Conv2d(3, 64, 4, 2, 1, bias = False)
			# LeakyReLU param -> negative_slope, inplace | negative_slope control the angle of the negative slope
			nn.LeakyReLU(0.2, inplace = True)
			nn.Conv2d(64, 128, 4, 2, 1, bias = False)
			nn.BatchNorm(128)
			nn.LeakyReLU(0.2, inplace = True)
			nn.Conv2d(128, 256, 4, 2, 1, bias = False)
			nn.BatchNorm(256)
			nn.LeakyReLU(0.2, inplace = True)
			nn.Conv2d(256, 512, 4, 2, 1, bias = False)
			nn.BatchNorm(512)
			nn.LeakyReLU(0.2, inplace = True)
			nn.Conv2d(512, 1, 4, 1, 0, bias = False)
			nn.Sigmoid()
		)
		
	def forward(self,input):
		output = self.main(input)
		return output.view(-1)
		
#create Discriminator object
netDis = Discriminator()
netDis.apply(weights_init)

#Training both Generator and Discriminator
#create object that will measure the error
criterion = nn.BCELoss() # creates a criterion that measures the Binary Cross Entrophy between target and output
#create optimizer for both Generator and Discriminator
DisOptimizer = optim.Adam(netGen.parameters(), lr = 0.0002, betas = (0.5, 0, 0.999))
GenOptimizer = optim.Adam(netDis.parameters(), lr = 0.0002, betas = (0.5, 0, 0.999))

#iterate for 25 training cycle | 1 training cycle = 1 epoch
for epoch in range(25):
	for i, data in enumerate(dataloader, 0):
		# 1st Step -> Update the weight of neural network in Discriminator
		netDis.zero_grad() # set gradients to 0 with respect to the weight
		
		# train the discriminator with real image of the datasets
		real,_ = data
		input = Variable(real)
		target = Variable(torch.ones(input.size()[0])) # get the target
		output = netDis(input)
		errDis_real = criterion(output,target)
		
		# train the discriminator with fake image created by the generator
		noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) #make a random input vector
		# forward propagate this random input vector into the neural network of the generator to get some fake generated images.
		fake = netGen(noise)
		target = Variable(torch.zeros(input.size()[0]))
		# forward propagate the fake generated images into the neural network of the discriminator to get the prediction (0 and 1)
		output = netDis(fake.detach())
		errDis_fake = criterion(output,target)
		
		# backpropagation sum of error
		errDis = errDis_real + errDis_fake
		# backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
		errDis.backward()
		# apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.
		DisOptimizer.step()
		
		# 2nd Step -> Update the weight of neural network in Generator
		netGen.zero_grad()
		target = Variable(torch.ones(input.size()[0]))
		output = netGen(fake)
		errGen = criterion(output, target)
		errGen.backward()
		GenOptimizer.step()
		
		# 3rd Step -> Print loss, and also save real images and generated images of the minibatch every 100 steps
		print('[%d/%d] [%d/%d] Loss_Dis : %.4f Loss_Gen : %.4f' %(epoch, 25, i, len(dataloader), errDis.data[0], errGen.data[0]))
		if i % 100 == 0:
			vutils.save_image(real, '%s/real_samples.png' %"./results", normalize = True)
			fake = netGen(noise)
			vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' %("./results", epoch), normalize = True)