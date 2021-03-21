'''

A program that demonstrates neural network. This program recognize a 8x8 hand-writing input “image”.
This is absolutely a toy neural network. There are much much improvement can be made.

'''

import math
import random
import time

class NormalizedData:
    def __init__(self, aInputSize):
        self.input = [0 for i in range(aInputSize)]
        self.target = None

class NeuralNetwork:
    def __init__(self, aInputNeu, aOutputSet, aHidden=40, aLearnRate=0.1, aMinError=0.001, aTrain=0.6):
        self.TRAINING_DATA = None
        self.TESTING_DATA = None
        self.INPUT_NEU = aInputNeu          # number of input neurons
        self.HIDDEN = aHidden               # number of hidden neurons
        self.OUTPUT_NEU = len(aOutputSet)   # number of output neurons
        self.LEARN_RATE = aLearnRate        # learning rate
        self.MIN_ERROR = aMinError          # threshold of error
        self.TRAIN = aTrain                 # proportion of error for training
        self.VALIDAIO  = 1-self.TRAIN            # proportion of error for validation

        self.hidden_w = [[random.random()-0.5 for x in range(self.HIDDEN)] for y in range(self.INPUT_NEU)]     # weights of hidden units
        self.out_w = [[random.random()-0.5 for x in range(self.OUTPUT_NEU)] for y in range(self.HIDDEN)]       # weights of output units
        self.hidden_b = [random.random()-0.5 for x in range(self.HIDDEN)]                   # bias of hidden units
        self.out_b = [random.random()-0.5 for x in range(self.OUTPUT_NEU)]                  # bias of output units
        self.hidden = [random.random()-0.5 for x in range(self.HIDDEN)]                     # values of hidden units
        self.output = [random.random()-0.5 for x in range(self.OUTPUT_NEU)]                 # values of output units

        self.outputSet = aOutputSet
        self.cur_valid_acc = 0.0
        self.training_time = 0.0

    def activate_func(self, aInput):
        return 1 / (1 + math.exp(-1 * aInput))

    def learn_from_file(self, aFileName):
        fp = open(aFileName, "r")

        line = float(fp.readline())
        print("-- Start learning from a file --")
        print("Accuracy: " + str(line))

        line = fp.readline()
        layer = int(line[0:line.find("I")])
        print("Number of Layer: " + str(layer))

        arr = line[line.find("I"):].split()
        self.INPUT_NEU = int(arr[1])
        self.HIDDEN = int(arr[3])

        for i in range(self.HIDDEN):
            arr = fp.readline().split()
            self.hidden_b[i] = float(arr[0])
            for j in range(self.INPUT_NEU):
                self.hidden_w[j][i] = float(arr[j+1])

        line = fp.readline()    # H 30 O 10
        arr = line.split()
        self.HIDDEN= int(arr[1])
        self.OUTPUT_NEU = int(arr[3])

        for i in range(self.OUTPUT_NEU):
            arr = fp.readline().split()
            self.out_b[i] = float(arr[0])
            for j in range(self.HIDDEN):
                self.out_w[j][i] = float(arr[j+1])
        fp.close()
        print("-- Done --")
        return

    # return list of NormalizedData
    def get_input(self, filename):
        normalizedDataArr = []
        testcase = 0
        lineNumber = 0
        outputNeu = 0

        fp = open(filename, "r")

        for line in fp:
            lineNumber += 1
            if lineNumber == 1:
                a, b = line.split()
                testcase = int(a)
                outputNeu = int(b)
                print("Total data size:: " + str(testcase))
                print("Number of features: : " + str(outputNeu))
            else:
                data = NormalizedData(self.INPUT_NEU)
                arr = line.split()
                for j in range(len(arr) - 1):
                    data.input[j] = float(arr[j])
                data.target = arr[len(arr)-1]
                normalizedDataArr.append(data)
        fp.close

        return normalizedDataArr, testcase, self.INPUT_NEU

    def gen_output(self, filename, aCurValidAcc, aHiddenBias, aHiddenWeight, aOutBias, aOutWeight):
        print("Generating output file with accuracy: " + str(aCurValidAcc))
        layerCount = 1

        fp = open(filename, "w")
        fp.write(str(aCurValidAcc) + "\n")
        fp.write(str(layerCount))
        fp.write("I " + str(self.INPUT_NEU) + " H " + str(self.HIDDEN) + "\n")

        for j in range(self.HIDDEN):
            fp.write(str(aHiddenBias[j]))
            for i in range(self.INPUT_NEU):
                fp.write(" " + str(aHiddenWeight[i][j]))
            fp.write("\n")

        fp.write("H " + str(self.HIDDEN) + " O " + str(self.OUTPUT_NEU) + "\n")
        for j in range(self.OUTPUT_NEU):
            fp.write(str(aOutBias[j]))
            for i in range(self.HIDDEN):
                fp.write(" " + str(aOutWeight[i][j]))
            fp.write("\n")

        fp.close()
        return True

    def forward_pass(self, aData):
        # Calculate the value of each hidden neuron
        for j in range(self.HIDDEN):
            w_sum = self.hidden_b[j]
            for i in range(self.INPUT_NEU):
                w_sum += aData.input[i] * self.hidden_w[i][j]
            self.hidden[j] = self.activate_func(w_sum)

        # Calculate the value of each output neuron
        for j in range(self.OUTPUT_NEU):
            w_sum = self.out_b[j]
            for i in range(self.HIDDEN):
                w_sum += self.hidden[i] * self.out_w[i][j]
            self.output[j] = self.activate_func(w_sum)

        # Calculate the error of each data by comparing the target and the output values
        target_set = [0] * self.OUTPUT_NEU  # array to store the target value
        error = 0
        target_set[self.outputSet.index(aData.target)] = 1
        for i in range(self.OUTPUT_NEU):
            error += (math.pow((target_set[i] - self.output[i]), 2)) / 2

        return error, target_set

    def backword_pass(self, aData, aTargetSet):

        # store the change in weight
        d_hidden_w = [[0.0 for x in range(self.HIDDEN)] for y in range(self.INPUT_NEU)]
        d_out_w = [[0.0 for x in range(self.OUTPUT_NEU)] for y in range(self.HIDDEN)]

        # store the change in bias
        d_hidden_b = [0.0]*self.HIDDEN
        d_out_b = [0.0]*self.OUTPUT_NEU

        hidden_temp = 0.0
        out_temp = [0.0]*self.OUTPUT_NEU

        # Use the formula to calculate change of weight of each output neuron
        for j in range(self.OUTPUT_NEU):
            out_temp[j] = (aTargetSet[j] - self.output[j]) * (1 - self.output[j]) * self.output[j]
            for i in range(self.HIDDEN):
                d_out_w[i][j] = self.LEARN_RATE * out_temp[j] * self.hidden[i]
            d_out_b[j] = self.LEARN_RATE * out_temp[j]

        # Use the formula to calculate change of weight of each hidden neuron
        for j in range(self.HIDDEN):
            hidden_temp = 0.0
            for i in range(self.OUTPUT_NEU):
                hidden_temp += out_temp[i] * self.out_w[j][i]
            hidden_temp *= (1 - self.hidden[j]) * self.hidden[j]
            for i in range(self.INPUT_NEU):
                d_hidden_w[i][j] = self.LEARN_RATE * hidden_temp * aData.input[i]
            d_hidden_b[j] = self.LEARN_RATE * hidden_temp

        # Update the weight and bias of each output neuron
        for j in range(self.OUTPUT_NEU):
            for i in range(self.HIDDEN):
                self.out_w[i][j] += d_out_w[i][j]
                self.out_b[j] += d_out_b[j]

        # Update the weight and bias of each hidden neuron
        for j in range(self.HIDDEN):
            for i in range(self.INPUT_NEU):
                self.hidden_w[i][j] += d_hidden_w[i][j]
                self.hidden_b[j] += d_hidden_b[j]

    def save_nn_file(self, output_file):
        success = True
        if not self.gen_output(output_file, self.cur_valid_acc, self.hidden_b, self.hidden_w, self.out_b, self.out_w):
            print("Error when generating output file")
            success = False
        return success

    def learning(self, input_file):
        # for calculating stopping criterion
        cur_error = -1
        prev_error = -1

        # for performing early stopping
        cur_valid_acc = -1
        prev_valid_acc = -1

        print("Training...")
        data, testcase, self.INPUT_NEU = self.get_input(input_file)
        self.TRAINING_DATA = data[0:int(testcase * self.TRAIN)]
        self.TESTING_DATA = data[int(testcase * self.TRAIN):]
        while True:
            prev_error = cur_error
            cur_error = 0.0

            # Train the neural network by the training data set
            for i in range(int(testcase * self.TRAIN)):
                # Perform forward pass and calculate the sum of error of this iteration
                err, target_set = self.forward_pass(data[i])
                cur_error = cur_error + err
                # Perform backward pass
                self.backword_pass(data[i], target_set)
            # Calculate the average error
            cur_error /= int(testcase * self.TRAIN)

            prev_valid_acc = cur_valid_acc
            correct_recognition = 0

            # Perform validation for the validation set
            for i in range(int(testcase * self.TRAIN), testcase):
                # Perform forward pass
                err, target_set = self.forward_pass(data[i])
                correct = 1
                # if one of the output is wrong, correct = 0
                for j in range(self.OUTPUT_NEU):
                    if ((j == self.outputSet.index(data[i].target) and self.output[j] <= 0.5) or \
                        (j != self.outputSet.index(data[i].target) and self.output[j] > 0.5)):
                        correct = 0
                        break
                if correct == 1:
                    correct_recognition += 1

            # Calculate the validation accuracy by counting the percentage of correct recognition
            cur_valid_acc = float(correct_recognition) / (testcase - int(testcase * self.TRAIN))
            print("Classification accuracy: " + str(cur_valid_acc))

            # Early stopping if the validation accuracy decrease
            if cur_valid_acc - prev_valid_acc <= 0.0:
                print("**********")
                break
            # Stop training if the training error is small enough
            if prev_error == -1 or prev_error - cur_error > self.MIN_ERROR:
                break

        print("Completed!")

        self.cur_valid_acc = cur_valid_acc
        return cur_valid_acc

    # aData is a NormalizedData
    def check(self, aData):
        target_set = [0] * self.OUTPUT_NEU       # array to store the target value
        hidden = [0.0] * self.HIDDEN             # values of hidden units
        output = [0.0] * self.OUTPUT_NEU         # values of output units

        # Calculate the value of each hidden neuron
        for j in range(self.HIDDEN):
            w_sum = self.hidden_b[j]
            for i in range(self.INPUT_NEU):
                w_sum += aData.input[i] * self.hidden_w[i][j]
            hidden[j] = self.activate_func(w_sum)

        # Calculate the value of each output neuron
        for j in range(self.OUTPUT_NEU):
            w_sum = self.out_b[j]
            for i in range(self.HIDDEN):
                w_sum += hidden[i] * self.out_w[i][j]
            output[j] = self.activate_func(w_sum)

        # Calculate the error of each data by comparing the target and the output values
        max = 0
        ans = None
        for i in range(self.OUTPUT_NEU):
            if output[i] >= max:
                max = output[i]
                ans = self.outputSet[i]

        return ans, output



def main(datafile, classifierfile):

    node_count = (30,81)
    trail = 10
    nn_array = []
    best_nn = None


    for hidden_node_size in range(node_count[0], node_count[1], 1):
        for i in range(trail):
            print("--- ---------- ---")
            nn = NeuralNetwork(64, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], aHidden=hidden_node_size,aTrain=0.7)
            start_time = time.time()
            acc = nn.learning(datafile)
            nn.training_time = (time.time() - start_time)
            if best_nn is None:
                best_nn = nn
            else:
                if nn.cur_valid_acc > best_nn.cur_valid_acc:
                    best_nn = nn
            nn_array.append(nn)
            print("--- Number of Hidden Node: %d" % (nn.HIDDEN))
            print("--- %s seconds ---" % nn.training_time)
        best_nn.save_nn_file(classifierfile)
    print("Done")

    print("==================================")
    print("datasize|number of features|accuracy|number of hidden node|time")
    for nn in nn_array:
        print("%d|%d|%f|%d|%f" % (len(nn.TRAINING_DATA) + len(nn.TESTING_DATA),
              nn.INPUT_NEU,
              nn.cur_valid_acc,
              nn.HIDDEN,
              nn.training_time))




def check():
    nn = NeuralNetwork(64, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    nn.learn_from_file("classifier.nn")
    print("Try to identify 0,0,7,11,11,6,0,0,0,9,16,12,10,14,0,0,0,5,2,0,4,14,0,0,0,0,1,5,14,6,0,0,0,1,15,16,16,10,0,0,0,0,7,4,4,15,6,0,0,0,5,4,8,13,12,0,0,0,14,16,12,10,1,0")
    dd = NormalizedData(64)
    dd.input = [0, 0, 7, 11, 11, 6, 0, 0, 0, 9, 16, 12, 10, 14, 0, 0, 0, 5, 2, 0, 4, 14, 0, 0, 0, 0, 1, 5, 14, 6, 0, 0,
                0, 1, 15, 16, 16, 10, 0, 0, 0, 0, 7, 4, 4, 15, 6, 0, 0, 0, 5, 4, 8, 13, 12, 0, 0, 0, 14, 16, 12, 10, 1,
                0]
    prediction, poss_arr = nn.check(dd)
    print("Predict the character is " + prediction)
    for i in range(10):
        poss = poss_arr[i]
        print("Possibility of being " + str(i) + " is " + str(round(poss, 4)))

start_time = time.time()
check()
print("Checking time: "+ str(time.time() - start_time))

#main("data/norm-data.txt", "classifier.nn")
#main("data/___norm.txt", "classifier.nn")

