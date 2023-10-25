import math
import random
import csv
import json

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_sizes, output_layer_size, weight_init_range, bias_init_range, alpha):
        self.input_layer_size = input_layer_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_layer_size = output_layer_size

        self.layer_sizes = [input_layer_size] + hidden_layer_sizes + [output_layer_size]
        self.layer_count = len(self.layer_sizes)

        self.weight_init_range = weight_init_range
        self.bias_init_range = bias_init_range

        self.alpha = alpha

        self.reset()

        self.init_neurons()
    
    def get_empty_weight_arrays(self):
        return [[]] + [
            [
                [
                    0 for k in range(self.layer_sizes[i - 1])
                ]
                for j in range(self.layer_sizes[i])
            ]
            for i in range(1, self.layer_count)
        ]

    def get_empty_bias_arrays(self):
        return [[]] + [
            [
                0 for j in range(self.layer_sizes[i])
            ]
            for i in range(1, self.layer_count)
        ]

    def add_weight_arrays(self, w1, w2):
        r = []
        for i in range(len(w1)):
            a1 = []
            for j in range(len(w1[i])):
                a2 = []
                for k in range(len(w1[i][j])):
                    a2.append(w1[i][j][k] + w2[i][j][k])
                a1.append(a2)
            r.append(a1)
        return r

    def add_bias_arrays(self, b1, b2):
        r = []
        for i in range(len(b1)):
            a = []
            for j in range(len(b1[i])):
                a.append(b1[i][j] + b2[i][j])
            r.append(a)
        return r
    
    def mul_weight_array(self, w, m):
        r = []
        for i in range(len(w)):
            a1 = []
            for j in range(len(w[i])):
                a2 = []
                for k in range(len(w[i][j])):
                    a2.append(w[i][j][k] * m)
                a1.append(a2)
            r.append(a1)
        return r


    def mul_bias_array(self, b, m):
        r = []
        for i in range(len(b)):
            a = []
            for j in range(len(b[i])):
                a.append(b[i][j] * m)
            r.append(a)
        return r

    
    def init_neurons(self):

        # Biases and weights are indexed 1 = second layer (first hidden layer)

        # weights[layer_to][index_to][index_from]
        self.weights = [[]] + [
            [
                [
                    random.random() * self.weight_init_range * 2 - self.weight_init_range
                    for k in range(self.layer_sizes[i - 1])
                ]
                for j in range(self.layer_sizes[i])
            ]
            for i in range(1, self.layer_count)
        ]

        self.biases = [[]] + [
            [
                random.random() * self.bias_init_range * 2 - self.bias_init_range
                for j in range(self.layer_sizes[i])
            ]
            for i in range(1, self.layer_count)
        ]
    
    def reset(self):
        self.inputs = []
        self.activations = []
        self.costs = []
    
    def process(self, input_activations, compare):
        # Init activations array (for all neurons)
        self.activations = [input_activations] + list([] for i in range(self.layer_count - 1))
        self.z = [[]] + list([] for i in range(self.layer_count - 1))

        # Calculate activations, working from input to output
        # 1 for first hidden layer
        for layer in range(1, self.layer_count):
            for i in range(self.layer_sizes[layer]):
                #print(self.activations)
                z = sum(
                        self.activations[layer - 1][j] * self.weights[layer][i][j]
                        for j in range(self.layer_sizes[layer - 1])
                    ) + self.biases[layer][i]
                self.z[layer].append(z)
                self.activations[layer].append(sigmoid(z))
        
        output_layer = self.activations[-1]
        
        # Get costs
        self.costs = [0.5 * (compare[i] - output_layer[i]) ** 2 for i in range(len(output_layer))]
        self.total_cost = sum(self.costs)


    # The calculus part
    def get_derivatives(self, compare):
        """
        Everything based on z
        Return weight and bias changes
        Activation nudges used intermediately

        """


        db = list([] for i in range(self.layer_count))
        dw = [[]] + [
            list([] for j in range(self.layer_sizes[i]))
            for i in range(1, self.layer_count)
        ]

        da = list(
            #2 * 
            (self.activations[-1][i] - compare[i])
            for i in range(self.layer_sizes[-1])
        )

        for layer in range(self.layer_count - 1, 0, -1):
            dz = [
                self.activations[layer][i] * (1 - self.activations[layer][i]) * da[i]
                for i in range(self.layer_sizes[layer])
            ]
            #print(layer, dz)
            for i in range(self.layer_sizes[layer]):
                db[layer].append(dz[i])
                for j in range(self.layer_sizes[layer - 1]):
                    dw[layer][i].append(
                        self.activations[layer - 1][j] * dz[i]
                    )
            if layer > 1:
                da = list(
                    sum(
                        self.weights[layer][j][i] * dz[j]
                        for j in range(self.layer_sizes[layer])
                    )
                    for i in range(self.layer_sizes[layer - 1])
                )
                

        return {
            "weight": dw,
            "bias": db,
        }

    def apply_changes(self, weight_changes, bias_changes):
        self.weights = self.add_weight_arrays(self.weights, weight_changes)
        self.biases = self.add_bias_arrays(self.biases, bias_changes)
    
    def run_epoch(self, inputs, corrects, batch_size):
        length = len(inputs)

        i = 0
        for j in range(int(length / batch_size)):


            total_bias_nudges = self.get_empty_bias_arrays()
            total_weight_nudges = self.get_empty_weight_arrays()
            
            total_cost = 0
            
            for k in range(batch_size):
                input = inputs[i]
                correct = corrects[i]
                self.process(input, correct)
                total_cost += self.total_cost
                nudges = self.get_derivatives(correct)
                total_bias_nudges = self.add_bias_arrays(total_bias_nudges, nudges["bias"])
                total_weight_nudges = self.add_weight_arrays(total_weight_nudges, nudges["weight"])
                i += 1

            # take average of all nudges returned from the batch
            """for layer_i in range(1, self.layer_count):
                for neuron_i in range(self.layer_sizes[layer_i]):
                    total_bias_nudges[layer_i][neuron_i] *= -self.alpha / batch_size
                    for sending_neuron_i in range(self.layer_sizes[layer_i - 1]):
                        total_weight_nudges[layer_i][neuron_i][sending_neuron_i] *= -self.alpha / batch_size"""
            
            # Apply
            self.apply_changes(
                self.mul_weight_array(total_weight_nudges, -self.alpha / batch_size),
                self.mul_bias_array(total_bias_nudges, -self.alpha / batch_size)
            )
            #print(self.mul_bias_array(total_bias_nudges, -self.alpha / batch_size))
            print(" Batch", j, "done")
            l = total_cost / batch_size
            print(" ", "Avg. loss:", l)
    
    def test_accuracy(self, inputs, corrects):
        total_correct = 0
        for i in range(len(inputs)):
            correct = corrects[i]
            self.process(inputs[i], correct)

            output_activations = self.activations[-1]
            
            max_i = -1
            curr_max = 0
            for i in range(len(output_activations)):
                if(output_activations[i] > curr_max):
                    curr_max = output_activations[i]
                    max_i = i
            if max_i == correct.index(1):
                total_correct += 1
        
        return total_correct / len(inputs)


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


"""network = NeuralNetwork(2, [10, 20], 2, 1, 1, 0.1)
input = [0.42, 0.5]
correct = [0.85, 0.95]
for i in range(10):
    network.process(input, correct)
    print(network.activations[-1], network.total_cost)
    d = network.get_derivatives(correct)
    network.apply_changes(
        mul_weight_array(d["weight"], -network.alpha),
        mul_bias_array(d["bias"], -network.alpha)
    )

d = network.get_derivatives(correct)
print(d["bias"])
print(add_bias_arrays(d["bias"], network.get_empty_bias_arrays()))"""


#data = format_data_as_input(data)


# simulate one epoch:

index = 0

# Fix this data!!!

"""test = [list(float(int(i) / 255) for i in data[0][1:]), []]
test[1] = list(0 for i in range(10))
test[1][int(data[0][0])] = 1
network.process(test[0], test[1])
before = network.activations[-1].copy()
print(network.activations[-1])
nudges = network.get_derivatives(test[1])
network.apply_changes(
    nudges["weight"],
    nudges["bias"]
)
network.process(test[0], test[1])
after = network.activations[-1].copy()
print(network.activations[-1])
print(test[1])
print(list(after[i] - before[i] for i in range(len(after))))"""


training_data = [
    list(data[i][0] for i in range(36000)),
    list(data[i][1] for i in range(36000))
]
test_data = [
    list(data[i][0] for i in range(-6000, 0)),
    list(data[i][1] for i in range(-6000, 0))
]



# last time: ended around 0.47 avg loss, 15.7% accuracy
"""
for i in range(2, 20):
    network.alpha = 2 / i ** 1.3
    print("Starting epoch with learning rate", round(network.alpha, 4))
    network.run_epoch(training_data[0], training_data[1], 1000)
    print("Accuracy:", network.test_accuracy(test_data[0], test_data[1]))
"""

network = NeuralNetwork(784, [16, 16], 10, 1, 1, 0.05)

# to continue training from pre-trained network
print("Retrieving neural network state...")
with open("network_data.json", "r") as fp:
    data = json.load(fp)
    network.weights = data["weights"]
    network.biases = data["biases"]
    network.alpha = data["alpha"]

print("Starting...\n")

# Last run: ended around 0.41 avg loss, 33.73% accuracy, consistent improvement of
# ~0.5% accuracy rate improvement per epoch
last_accuracy = network.test_accuracy(test_data[0], test_data[1])
print("Initial accuracy:", round(last_accuracy, 5))

network.alpha = 0.03

while True:
    print("Starting epoch with learning rate", round(network.alpha, 4))
    network.run_epoch(training_data[0], training_data[1], 1000)
    a = network.test_accuracy(test_data[0], test_data[1])
    print("Accuracy:", round(a, 5), "|", "Improvement:", round(a - last_accuracy, 5))
    last_accuracy = a
    with open("network_data.json", "w") as jsonFile:
        json.dump(network.__dict__, jsonFile)

"""

# Test accuracy:
correct = 0
total = 0
while total < len(test_data):

"""