from math import exp
from random import random, randint
import numpy as np
import csv
import json


# activation functions
def sig(x):
  return 1 / (1 + exp(-x))

def d_sig(x):
  s = sig(x)
  return s * (1 - s)

def relu(x):
  return max(0.0, x)

def d_relu(x):
  return x * (x > 0)


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
    self.w = [[]] + [
      [
        [(2 * random() - 1) * w_range for k in range(self.layer_sizes[i - 1])]
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]

    self.b = [[]] + [
      [(2 * random() - 1) * b_range for j in range(self.layer_sizes[i])]
      for i in range(1, self.layer_count)
    ]
  

  def feed_forward(self, inputs):
    self.z = [[]]
    self.a = [inputs]

    for l in range(1, self.layer_count):
      z_row = [
        sum(
          self.a[l - 1][j] * self.w[l][i][j]
          for j in range(self.layer_sizes[l - 1])
        ) + self.b[l][i]
        for i in range(self.layer_sizes[l])
      ]
      self.z.append(z_row)
      self.a.append([self.activation_func(i) for i in z_row])
  

  def get_cost(self, expected):
    return sum((expected[i] - self.a[-1][i]) ** 2 for i in range(self.layer_sizes[-1]))
  

  def calculate_gradient(self, expected):
    ll_da = [-2 * (expected[i] - self.a[-1][i]) for i in range(self.layer_sizes[-1])]
    da = [[] for i in range(self.layer_count - 1)] + [ll_da]

    ll_dz = [
      ll_da[i] * self.activation_func_derivative(self.z[-1][i])
      for i in range(self.layer_sizes[-1])
    ]
    dz = [[] for i in range(self.layer_count - 1)] + [ll_dz]
    
    db = [[] for i in range(self.layer_count - 1)] + [ll_dz]

    ll_dw = [
      [
        ll_dz[i] * self.a[-2][j] for j in range(self.layer_sizes[-2])
      ]
      for i in range(self.layer_sizes[-1])
    ]
    dw = [[] for i in range(self.layer_count - 1)] + [ll_dw]

    for layer in range(self.layer_count - 2, 0, -1):
      l_da = [
        sum(
          dz[layer + 1][j] * self.w[layer + 1][j][i]
          for j in range(self.layer_sizes[layer + 1])
        )
        for i in range(self.layer_sizes[layer])
      ]
      da[layer] = l_da

      l_dz = [
        l_da[i] * self.activation_func_derivative(self.z[layer][i])
        for i in range(self.layer_sizes[layer])
      ]
      dz[layer] = l_dz

      l_dw = [
        [
          dz[layer][i] * self.a[layer - 1][j]
          for j in range(self.layer_sizes[layer - 1])
        ]
        for i in range(self.layer_sizes[layer])
      ]
      dw[layer] = l_dw

      db[layer] = l_dz

    
    return {
      "z": dz,
      "a": da,
      "w": dw,
      "b": db,
    }


  def get_zero_weight_array(self):
    return [[]] + [
      [
        [0] * self.layer_sizes[i - 1]
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]
  

  def get_zero_bias_array(self):
    return [[]] + [
      [0] * self.layer_sizes[i]
      for i in range(1, self.layer_count)
    ]
  

  def add_weight_arrays(self, w1, w2):
    return [[]] + [
      [
        [w1[i][j][k] + w2[i][j][k] for k in range(self.layer_sizes[i - 1])]
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]
  

  def add_bias_arrays(self, b1, b2):
    return [[]] + [
      [
        b1[i][j] + b2[i][j]
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]
  

  def mul_weight_by_scalar(self, w, s):
    return [[]] + [
      [
        [
          w[i][j][k] * s
          for k in range(self.layer_sizes[i - 1])
        ]
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]
  

  def mul_bias_by_scalar(self, b, s):
    return [[]] + [
      [
        b[i][j] * s
        for j in range(self.layer_sizes[i])
      ]
      for i in range(1, self.layer_count)
    ]


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

training_data = data[:30000]
test_data = data[-6000:]


net = DenselyConnectedNeuralNetwork([784, 30, 10], sig, d_sig)

restart = False

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


epoch_length = 10
batch_size = 3000

alpha = 0.1

epoch = net.epochs + 1

while True:
  print("Starting epoch", str(epoch) + "...")
  data_index = 0
  for i in range(epoch_length):
    cost_running_total = 0
    total_correct = 0
    weight_nudges_total = net.get_zero_weight_array()
    bias_nudges_total = net.get_zero_bias_array()
    for j in range(batch_size):
      inputs = training_data[data_index][0]
      expected = training_data[data_index][1]

      net.feed_forward(inputs)

      ll_a = net.a[-1]
      answer = [0] * 10
      answer[ll_a.index(max(ll_a))] = 1
      if answer == expected:
        total_correct += 1

      cost = net.get_cost(expected)
      cost_running_total += cost

      grad = net.calculate_gradient(expected)
      
      dw = grad["w"]
      db = grad["b"]

      weight_nudges_total = net.add_weight_arrays(weight_nudges_total, dw)
      bias_nudges_total = net.add_bias_arrays(bias_nudges_total, db)

      data_index += 1
    
    weight_nudges_total = net.mul_weight_by_scalar(weight_nudges_total, -alpha / batch_size)
    bias_nudges_total = net.mul_bias_by_scalar(bias_nudges_total, -alpha / batch_size)

    net.w = net.add_weight_arrays(net.w, weight_nudges_total)
    net.b = net.add_bias_arrays(net.b, bias_nudges_total)

    print(
      str(i + 1) + ".", total_correct, "/", batch_size, ",",
      round(total_correct / batch_size, 4),
      "|", "Avg. cost", cost_running_total / batch_size
    )
  epoch += 1
  net.epochs += 1

  # Store state after each epoch
  with open("network_data.json", "w") as jsonFile:
    d = dict(net.__dict__)
    del d["activation_func"]
    del d["activation_func_derivative"]
    json.dump(d, jsonFile)