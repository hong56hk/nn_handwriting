'''

A program that demonstrates using k-nearest neighbors algorithm to recognize a 8x8 hand-writing.
This program recognize a 8x8 hand-writing input “image”.
This is absolutely a toy neural network. There are much much improvement can be made.

'''

import math
import operator
import time

# data object for nn
class NormalizedData:
    def __init__(self, aInputSize):
        self.input = [0 for i in range(aInputSize)]
        self.target = None

class KNN:

    #
    # k: the nubmer of possible prediction, ie number of group
    # group_key: the possible prediction
    # splite is the ratio of training:testing from the input data
    def __init__(self, k, attribute_size, group_key, split=0.67):
        self.training_data_array = []
        self.k = k
        self.training_data_size = 0
        self.attribute_size = attribute_size
        self.group_center = {}
        self.group_key = group_key

        self.split = split
        self.testing_data_array = [] # in NormalizedData format
        self.testing_data_size = 0

    # return list of NormalizedData
    def get_input(self, filename):
        normalizedDataArr = []
        testcase = 0
        lineNumber = 0
        attribute_size = 0

        fp = open(filename, "r")

        for line in fp:
            lineNumber += 1
            if lineNumber == 1:
                a, b = line.split()
                testcase = int(a)
                attribute_size = int(b)
                print("Number of data: " + str(testcase))
                print("Number of attributes: " + str(attribute_size))
                print("Reading input file")
            else:
                data = NormalizedData(attribute_size)
                arr = line.split()
                for j in range(len(arr) - 1):
                    data.input[j] = float(arr[j])
                data.target = arr[len(arr) - 1]
                normalizedDataArr.append(data)
        fp.close

        return normalizedDataArr, testcase, attribute_size



    def learn_from_file(self, file):
        print("Learning from the file: " + file)
        pass

    def learning(self, datafile, resultfile):
        normalizedDataArr, data_size, self.attribute_size = self.get_input(datafile)

        self.training_data_size = round(self.split*data_size)
        self.testing_data_size = round((1-self.split) * data_size)

        self.training_data_array = normalizedDataArr[0:self.training_data_size-1]
        self.testing_data_array = normalizedDataArr[self.testing_data_size:]

        print("Start learning the file")
        print("Traning data size: " + str(self.training_data_size))
        print("Testing data size: " + str(self.testing_data_size))
        print("Number of features: " + str(self.attribute_size))

        # init the training data dictionary
        # key is the result; value is the feature
        traindata = {}
        for grp_key in self.group_key:
            traindata[grp_key] = []

        for ndata in self.training_data_array:
            traindata[ndata.target].append(ndata)

        # calculate the center of each group of data
        for grp_key in traindata.keys():
            group_center_tmp = [0 for i in range(self.attribute_size)]
            for ndata in traindata[grp_key]:
                for i in range(self.attribute_size):
                    group_center_tmp[i] += ndata.input[i]
            for i in range(self.attribute_size):
                group_center_tmp[i] = group_center_tmp[i]/len(traindata[grp_key])
            self.group_center[grp_key] = group_center_tmp

        # output the group center to file
        # [total number of training data] [total number of the group]
        # [key] [number of data] [center of feature0] [center of feature1] [...]
        resultfile = open(resultfile, "w")
        resultfile.write(str(self.training_data_size) + " " + str(self.attribute_size) + "\n")
        for grp_key in traindata.keys():
            grp_data = traindata[grp_key]
            resultfile.write(str(grp_key) + " " + str(len(grp_data )) + " ")
            for i in range(self.attribute_size-1):
                resultfile.write(str(self.group_center[grp_key][i]) + " ")
            resultfile.write(str(self.group_center[grp_key][-1]) + "\n")
        resultfile.close()

        # calculate the accuracy
        correct_count = 0
        incorrect_count = 0
        for i in range(self.testing_data_size):
            ndata = self.testing_data_array[i]
            prediction, summary = self.predict(ndata)
            if prediction == ndata.target:
                correct_count += 1
            else:
                incorrect_count += 1
        print("Number of testing data: " + str(self.testing_data_size))
        print("Nubmer of identifing the character correctly: " + str(correct_count))
        print("Accuracy: " + str(round(correct_count*100.0/self.testing_data_size,2)))

        return True

    # newdata should be a NormalizedData while the target is None
    def predict(self, newdata):
        distance = {}
        # calculate the euclidean distance between the newdata and all the existing groups
        for grp_key in self.group_center.keys():
            grp_center = self.group_center[grp_key]
            distance[grp_key] = self.euclid_dist(grp_center, newdata.input)

        # sort the distance in ascending order
        # the shortest distance the prediction
        sorted_distance = sorted(distance.items(), key=operator.itemgetter(1))

        return sorted_distance[0][0], sorted_distance

    # print the details distance
    # summary is the output from predict function
    def print_summary(self, summary):
        print("Predict the character is " + str(summary[0][0]))
        for dd in summary:
            prediction, dist = dd
            print("Distance between " + str(prediction) + " is " + str(round(dist,4)))

    # instane1 and instance2 are array of attribute
    # [attribute0, attribute1, attribute2, ..., attributeN]
    def euclid_dist(self, instance1, instance2):
        distance = 0
        if len(instance1) == len(instance2):
            for i in range(len(instance1)):
                distance += pow((instance1[i] - instance2[i]), 2)
            distance = math.sqrt(distance)
        else:
            distance = -1
        return distance



def main(datafile, classifierfile):


    knn = KNN(10, 64, ['0','1','2','3','4','5','6','7','8','9'])

    print("Starting to learn")
    start_time = time.time()
    knn.learning(datafile, classifierfile)
    print("Done")
    print("--- %s seconds ---" % (time.time() - start_time))


    print("Try to identify 0,0,7,11,11,6,0,0,0,9,16,12,10,14,0,0,0,5,2,0,4,14,0,0,0,0,1,5,14,6,0,0,0,1,15,16,16,10,0,0,0,0,7,4,4,15,6,0,0,0,5,4,8,13,12,0,0,0,14,16,12,10,1,0")
    dd = NormalizedData(64)
    dd.input = [0,0,7,11,11,6,0,0,0,9,16,12,10,14,0,0,0,5,2,0,4,14,0,0,0,0,1,5,14,6,0,0,0,1,15,16,16,10,0,0,0,0,7,4,4,15,6,0,0,0,5,4,8,13,12,0,0,0,14,16,12,10,1,0]
    prediction, summary = knn.predict(dd)
    print(prediction)
    knn.print_summary(summary)


main('data/norm-data.txt', 'classifier.knn')