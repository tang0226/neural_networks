from math import exp
from random import random, randint, shuffle
import numpy as np
import csv
import json


# activation functions, must be matrix-compatible
def sig(x):
  return 1 / (1 + np.exp(-x))

def d_sig(x):
  s = sig(x)
  return s * (1 - s)

# Fix these
"""def relu(x):
  return max(0.0, x)

def d_relu(x):
  return x * (x > 0)"""

def neg_arr(arr):
  return [-i for i in arr]

def add_arr(a1, a2):
  return [a1[i] + a2[i] for i in range(len(a1))]

def mul_arr(a, s):
  return [i * s for i in a]


# Start with vanilla, move to np later
class DenselyConnectedNeuralNetwork:
  def __init__(self, layer_sizes, activation_func, activation_func_derivative):
    self.layer_sizes = layer_sizes
    self.ll_size = layer_sizes[-1]

    self.layer_count = len(layer_sizes)
    
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative
    
    self.w = []
    self.b = []

    self.epochs = 0
  

  def init_weights_and_biases(self, w_range, b_range):
    self.w = [
      np.random.rand(self.layer_sizes[i], self.layer_sizes[i - 1]) * (w_range * random() - 1)
      for i in range(1, self.layer_count)
    ]

    self.b = [
      np.random.rand(self.layer_sizes[i]) * (b_range * random() - 1)
      for i in range(1, self.layer_count)
    ]
  

  def feed_forward(self, inputs):
    self.z = []
    self.a = [np.array(inputs)]

    for l in range(self.layer_count - 1):
      z_row = np.dot(self.w[l], self.a[l]) + self.b[l]
      self.z.append(z_row)
      self.a.append(self.activation_func(z_row))
  

  def get_cost(self, expected):
    return np.sum((expected - self.a[-1]) ** 2)
  

  def calculate_gradient(self, expected):
    ll_da = -2 * (expected - self.a[-1])
    da = [0] * (self.layer_count - 2) + [ll_da]

    ll_dz = np.multiply(ll_da, self.activation_func_derivative(self.z[-1]))
    dz = [0] * (self.layer_count - 2) + [ll_dz]

    ll_dw = np.outer(ll_dz, self.a[-2])
    dw = [0] * (self.layer_count - 2) + [ll_dw]
    
    for layer in range(self.layer_count - 3, -1, -1):
      da[layer] = np.dot(np.transpose(self.w[layer + 1]), dz[layer + 1])
      dz[layer] = np.multiply(da[layer], self.activation_func_derivative(self.z[layer]))
      dw[layer] = np.outer(dz[layer], self.a[layer])

    
    return {
      "w": dw,
      "b": dz, # dz/db = 1, dC/db = dC/dz * dz/db = dC/dz, so dC/db = dC/dz
    }
  
  def apply_changes(self, w, b):
    for i in range(self.layer_count - 1):
      self.w[i] += w[i]
      self.b[i] += b[i]


  def get_zero_weight_array(self):
    return [
      np.zeros([self.layer_sizes[i], self.layer_sizes[i - 1]])
      for i in range(1, self.layer_count)
    ]
  

  def get_zero_bias_array(self):
    return [
      np.zeros(self.layer_sizes[i])
      for i in range(1, self.layer_count)
    ]



"""net = DenselyConnectedNeuralNetwork([4, 3, 2], sig, d_sig)
net.init_weights_and_biases(2, 2)
net.feed_forward([0.05, 0.1, 0.5, 0.2])
print(net.get_cost([1, 0]))
print()
g = net.calculate_gradient([1, 0])
print(g)
print(net.w, net.b)
net.apply_changes(mul_arr(g["w"], -1), mul_arr(g["b"], -1))
net.feed_forward([0.05, 0.1, 0.5, 0.2])
print(net.get_cost([1, 0]))"""



def format_data_as_input(data):
  digs = list(i[0] for i in data)
  corrects = list(list(0 for i in range(10)) for j in data)
  for i in range(len(data)):
    corrects[i][int(digs[i])] = 1
  inputs = list(
    list(int(j) / 255 for j in i[1:])
    for i in data
  )
  return [inputs, corrects]



# init data
print("\nInitializing training and testing data...")
with open("train.csv") as csvfile:
  read = csv.reader(csvfile, delimiter=",")
  data = list(read)[1:]

for i in range(len(data)):
  to_set = [list(int(j) / 255 for j in data[i][1:]), list(0 for j in range(10))]
  to_set[1][int(data[i][0])] = 1
  data[i] = to_set



# Configuration:

net = DenselyConnectedNeuralNetwork([784, 30, 10], sig, d_sig)

restart = True

alpha = 0.3
epoch_length = 1000
batch_size = 32
batch_print_interval = 100

training_data_count = 32000
testing_data_count = 8400



training_data_temp = data[:training_data_count]
training_data = list([i[0].copy(), i[1].copy()] for i in training_data_temp)

testing_data = data[-testing_data_count:]

if restart:
  # Get fresh random state
  net.init_weights_and_biases(2, 2)
else:
  # Import state from JSON storage
  print("Retrieving neural network state...")
  with open("network_data.json", "r") as fp:
    data = json.load(fp)
    net.w = data["w"]
    net.b = data["b"]
    net.epochs = data["epochs"]

print()



epoch = net.epochs + 1

while True:
  print("Starting epoch", str(epoch) + ",", "LR =", str(alpha) + "...")

  shuffle(training_data)

  data_index = 0
  epoch_cost_running_total = 0
  epoch_total_correct = 0

  for i in range(epoch_length):
    print_batch = (i + 1) % batch_print_interval == 0

    if print_batch:
      batch_cost_running_total = 0
    
    weight_nudges_total = net.get_zero_weight_array()
    bias_nudges_total = net.get_zero_bias_array()

    for j in range(batch_size):
      inputs = np.array(training_data[data_index][0])
      expected = np.array(training_data[data_index][1])

      net.feed_forward(inputs)

      ll_a = list(net.a[-1])
      answer = [0] * 10
      answer[list(ll_a).index(max(list(ll_a)))] = 1
      if answer == list(expected):
        epoch_total_correct += 1

      cost = net.get_cost(expected)

      if print_batch:
        batch_cost_running_total += cost
      
      epoch_cost_running_total += cost

      grad = net.calculate_gradient(expected)
      
      dw = grad["w"]
      db = grad["b"]

      weight_nudges_total = add_arr(weight_nudges_total, dw)
      bias_nudges_total = add_arr(bias_nudges_total, db)

      data_index += 1
    
    weight_nudges_total = mul_arr(weight_nudges_total, -alpha / batch_size)
    bias_nudges_total = mul_arr(bias_nudges_total, -alpha / batch_size)

    net.w = add_arr(net.w, weight_nudges_total)
    net.b = add_arr(net.b, bias_nudges_total)

    if print_batch:
      print(" Batch", str(i + 1) + ":", "Avg. cost", batch_cost_running_total / batch_size)
  

  # validation
  num_correct = 0
  total_cost = 0
  for data in testing_data:
    inputs = np.array(data[0])
    expected = data[1]

    net.feed_forward(inputs)

    ll_a = list(net.a[-1])
    answer = [0] * 10
    answer[list(ll_a).index(max(list(ll_a)))] = 1
    if answer == expected:
      num_correct += 1
    total_cost += net.get_cost(expected)

  print("Epoch", epoch, "done")
  print(
    " Cost:", round(epoch_cost_running_total / (batch_size * epoch_length), 5),
    "(" + str(round(total_cost / testing_data_count, 5)) + ")"
  )
  print(
    " Accuracy:",
    str(epoch_total_correct) + "/" + str(batch_size * epoch_length),
    "(" + str(num_correct) + "/" + str(testing_data_count) + ")" + ",",

    round(epoch_total_correct / (batch_size * epoch_length), 5),
    "(" + str(round(num_correct / testing_data_count, 5)) + ")"
  )
  print()
  epoch += 1
  net.epochs += 1

  # Store state after each epoch
  with open("network_data.json", "w") as jsonFile:
    d = dict(net.__dict__)
    del d["activation_func"]
    del d["activation_func_derivative"]
    for k in ["z", "a", "w", "b"]:
      d[k] = [i.tolist() for i in d[k]]
    json.dump(d, jsonFile)