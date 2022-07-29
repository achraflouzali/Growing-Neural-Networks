import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, input_dim = 10, hidden_dim = 5, output_dim = 2, new_neurons = 5):
      super(Net, self).__init__()
      self.input_dim = input_dim
      self.hidden_dim = hidden_dim
      self.output_dim = output_dim
      self.new_neurons = new_neurons
      self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
      self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def add_dimensions_cat(self, number_of_dim=5):
        new_neurons_in = torch.nn.Parameter(torch.rand(number_of_dim,10), requires_grad=True)
        self.fc1.weight = torch.nn.Parameter(torch.cat([new_neurons_in,new_neurons_in]))

        bias_in  = self.fc1.bias.data
        new_neurons_in_bias = torch.nn.Parameter(torch.rand(number_of_dim), requires_grad=True)
        self.fc1.bias = torch.nn.Parameter(torch.cat([bias_in,new_neurons_in_bias]))

        weight_out = self.fc2.weight.data
        new_neurons_out = torch.nn.Parameter(torch.rand(2,number_of_dim), requires_grad=True)
        self.fc2.weight = torch.nn.Parameter(torch.cat([weight_out,new_neurons_out], dim=1))
        

    def add_dimensions(self):
        weight_in  = self.fc1.weight.data
        bias_in = self.fc1.bias.data
        weight_out = self.fc2.weight.data
        bias_out = self.fc2.bias.data

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim+self.new_neurons)
        self.fc2 = nn.Linear(self.hidden_dim+self.new_neurons, self.output_dim)

        self.fc1.weight.data[:self.hidden_dim] = weight_in
        self.fc1.bias.data[:self.hidden_dim] = bias_in
        self.fc2.weight.data[:,:self.hidden_dim] = weight_out
        self.fc2.bias.data = bias_out

    # x represents our data
    def forward(self, x):
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

D_in, D_out = 10, 2

Exemplesx = []
Exemplesy = []
for i in range (0,200):
    x = Variable(torch.randn(1, D_in))
    y = Variable(torch.randn(D_out, 1))
    Exemplesx.append(x)
    Exemplesy.append(y)

# define random data
random_input = Variable(torch.randn(1,10))
random_target = Variable(torch.randn(2,1))

my_nn = Net()
my_nn.add_dimensions()

criterion = nn.MSELoss()
optimizer = optim.SGD(my_nn.parameters(), lr=0.1)


for i in range(5):
    for j in range(len(Exemplesx)):
        my_nn.zero_grad()
        output = my_nn(Exemplesx[j])
        loss = criterion(output, Exemplesy[j])
        loss.backward()
        my_nn.fc1.weight.grad[:my_nn.hidden_dim] = 0
        my_nn.fc1.bias.grad[:my_nn.hidden_dim] = 0
        my_nn.fc2.weight.grad[:,:my_nn.hidden_dim] = 0
        my_nn.fc2.bias.grad[:] = 0
        
        optimizer.step()

