"""
Adeena Ahmed, Suada Demirovic, Yash Dhayal, Jason Swick
CSC 426-01
File Name: network.py
Final Project
Description: Implementation of a multilayer neural network learner that utilizes backpropagation.

Structural Description:
    The NN class is used to create the neural network and all of its functionalities. The __init__ function
    initializes the neural network by taking in the number of input, hidden, and output units as parameters.
    The function sets the learning rate to 0.3 and the weights to random values in the interval (-0.1, 0.1).

    The sigmoid function takes in the dot product as a parameter and is called by the forward function to 
    propagate the input forward through the network. It returns a calculation of the sigmoid function.

    The SSE_plot function takes in the epoch_counter as a parameter in order to produce a plot of the sum of
    squared errors vs. the epoch number. This is written to the SquaredErrorsPlot.png file.

    The HUE_plot function takes in the filename of a hidden unit encoding file and epoch_counter to produce a
    plot of the hidden unit value vs. the epoch number. This is written to a HiddenUnit_NUMBER_Plot.png file, 
    where NUMBER corresponds to the input value.

    The forward function takes in data as a parameter, which is the training instance, and returns a list of
    outputs from the hidden and output units. It is called by the backpropagate function to propagate the input
    forward through the network.

    The backpropagate function takes in the parameters data, which is a list of training instances, and epochs,
    which is the number of epochs. It returns a Sum of Squared Error csv file with a column for each output unit
    and Hidden Unit Encoding files for each input.

    The main function takes no parameters, initializes the training data and neural network, and runs the 
    backpropagate function.
"""

from random import random
import matplotlib.pyplot as plt
import csv
import numpy
import math

class NN: 
    def __init__(self,inputs,hidden,outputs,lr = 0.3,weights = None):
        """
        Arguments
	    -------------------------------------------------------------------- 
            inputs: int : number of input units
            hidden: int : number of hidden units
            inputs: int : number of output units
            lr : float : learning rate
            weights: list: list of weights for each hidden unit & output unit
        """

        self.lr = lr # The learning rate is initialized to 0.3 by default.
        self.n_in = inputs
        self.n_h = hidden
        self.n_out = outputs
        self.w0 = 1   # Initializes a constant w0 weight.

        if weights is None: # Initializes weights to random values in the interval (-0.1, 0.1).
            self.weights = [[random()*0.2 - 0.1 for i in range(self.n_in + 1)] for j in range(self.n_h)]
            self.weights.extend([[random()*0.2 - 0.1 for i in range(self.n_h + 1)] for j in range(self.n_out)])
        else:
            self.weights = weights
 

    # Calculates the sigmoid function using the mathematical definition.
    # Used to propagate the input forward.
    def sigmoid(self, x):
	"""
        Arguments
	    -------------------------------------------------------------------- 
            x: the argument of sigmoid function
        """
        return float( 1 / ( 1 + ( math.e**(-1*x) ) ) )


    # Creates a plot of the sum of squared errors vs. the epoch number.
#     def SSE_plot(self, epoch_counter):
# 	"""
#         Arguments
# 	    -------------------------------------------------------------------- 
#             epoch_counter: number of epochs to be plotted 
#         """
#         # Initializes the units, as values corresponding to the unit's SSEs will be plotted on the y axis.
#         unit1,unit2,unit3,unit4,unit5,unit6,unit7,unit8 = [],[],[],[],[],[],[],[]

#         with open("SumOfSquaredErrors.csv", "r") as csvfile:
#             csvreader = csv.reader(csvfile)
#             next(csvreader, None) # Skips the first row of the csv file, as this is the header.

#             # Appends each column to an array corresponding to the specific unit's values.
#             for line in csvreader:
#                 unit1.append(float(line[0]))
#                 unit2.append(float(line[1]))
#                 unit3.append(float(line[2]))
#                 unit4.append(float(line[3]))
#                 unit5.append(float(line[4]))
#                 unit6.append(float(line[5]))
#                 unit7.append(float(line[6]))
#                 unit8.append(float(line[7]))

#         # Plots data corresponding to each of the 8 units on one graph.
#         fig = plt.figure(figsize=(12,8))
#         plt.plot(epoch_counter, unit1, label = "Unit 1")
#         plt.plot(epoch_counter, unit2, label = "Unit 2")
#         plt.plot(epoch_counter, unit3, label = "Unit 3")
#         plt.plot(epoch_counter, unit4, label = "Unit 4")
#         plt.plot(epoch_counter, unit5, label = "Unit 5")
#         plt.plot(epoch_counter, unit6, label = "Unit 6")
#         plt.plot(epoch_counter, unit7, label = "Unit 7")
#         plt.plot(epoch_counter, unit8, label = "Unit 8")
#         plt.legend(loc="center right")
#         plt.xlabel('Epoch Number')
#         plt.ylabel('Sum of Squared Errors')
#         plt.title('Sum of Squared Errors For Each Output Unit')

#         fig.tight_layout()
#         fig.savefig('SquaredErrorsPlot.png') # Saves the graph as a png file.


    # Creates a plot of the hidden unit value vs. the epoch number.
    # The filename passed in is used to distinguish between the different HUEs.
#     def HUE_plot(self, filename, epoch_counter):
# 	"""
#         Arguments
# 	    -------------------------------------------------------------------- 
#             filename: the Hidden Unit Encoding file whose data is to be plotted
# 	    epoch_counter: number of epochs to be plotted 
#         """
#         # Initializes the units, as their values will be plotted on the y axis.
#         unit1,unit2,unit3 = [],[],[]

#         with open(filename, "r") as csvfile:
#             csvreader = csv.reader(csvfile)
#             next(csvreader, None) # Skips the first row of the csv file, as this is the header.

#             # Appends values corresponding to each unit to an array.
#             for line in csvreader:
#                 unit1.append(float(line[0]))
#                 unit2.append(float(line[1]))
#                 unit3.append(float(line[2]))

#         # Plots data corresponding to each of the hidden units on one graph.
#         fig = plt.figure(figsize=(12,8))
#         plt.plot(epoch_counter, unit1, label = "Unit 1")
#         plt.plot(epoch_counter, unit2, label = "Unit 2")
#         plt.plot(epoch_counter, unit3, label = "Unit 3")
#         plt.legend(loc="center right")
#         plt.xlabel('Epoch Number')
#         plt.ylabel('Hidden Unit Value')  

#         # Sets titles corresponding to the specific HUE.
#         # Saves graphs to different files corresponding to the HUE.
#         if (filename == "HiddenUnitEncoding_10000000.csv"):
#             plt.title('Hidden Unit Encoding for Input 10000000')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_10000000_Plot.png')
#         elif (filename == "HiddenUnitEncoding_01000000.csv"):
#             plt.title('Hidden Unit Encoding for Input 01000000')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_01000000_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00100000.csv"):
#             plt.title('Hidden Unit Encoding for Input 00100000')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00100000_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00010000.csv"):
#             plt.title('Hidden Unit Encoding for Input 00010000')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00010000_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00001000.csv"):
#             plt.title('Hidden Unit Encoding for Input 00001000')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00001000_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00000100.csv"):
#             plt.title('Hidden Unit Encoding for Input 00000100')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00000100_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00000010.csv"):
#             plt.title('Hidden Unit Encoding for Input 00000010')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00000010_Plot.png')
#         elif (filename == "HiddenUnitEncoding_00000001.csv"):
#             plt.title('Hidden Unit Encoding for Input 00000001')
#             fig.tight_layout()
#             fig.savefig('HiddenUnit_00000001_Plot.png')


    def forward(self, data):
        """
	    Performs forward propagation in network 
	    Arguments
	    --------------------------------------------------------------------
		    data : list : training instance containing inputs
        Returns:      
	    --------------------------------------------------------------------
            output_h : list : list of outputs from hidden units 
            output_k : list : list of outputs from output units
	    """

        # Initializes hidden units & weights, and output units & weights.
        h_output = [1,0,0,0] # The result of hidden units after utilizing input units.
        h_weights = self.weights[:self.n_h]      
        
        
        k_output = [0,0,0,0,0,0,0,0] # The result of output units after utilizing hidden units.
        k_weights = self.weights[self.n_h:]      

        for h in range(1, len(h_output)): # Takes input units into hidden units.
            dotprod = numpy.dot(h_weights[h-1], data)  
            h_output[h] = self.sigmoid(dotprod)

        for k in range(len(k_output)): # Takes hidden units as inputs for output units.
            dotprod = numpy.dot(k_weights[k], h_output)
            k_output[k] = self.sigmoid(dotprod)
        return h_output, k_output
        

    def backpropagate(self,data,epochs=4999): # The epoch number is set to 4999 since they start at index 0. This will end up being 5000 epochs total. 
        """
	    Performs backward propagation in network for specified number of epochs
	    Arguments
	    --------------------------------------------------------------------
		    data : list : list of training instances  
            	    epochs: number of epochs or iterations
	
	    --------------------------------------------------------------------
	    Outputs:
            1. a SSE (Sum of Squared Error) csv file with a column for each output unit
            2. Hidden Unit Encoding file(s) for each input that contains a column for each hidden unit 
	    """
        n_epochs = 0
        epoch_counter = [] # An array stores epoch numbers, which will be used for the x axis when plotting.

        h_weights = self.weights[:self.n_h]
        k_weights = self.weights[self.n_h:]

        writers = []

        # Writes the sum of squared errors values for each of the output units to a csv file. 
        SSE_file = open("SumOfSquaredErrors.csv", 'w', newline='')
        write_SSE = csv.writer(SSE_file)
        write_SSE.writerow(["SSEoutputunit1,", "SSEoutputunit2,", "SSEoutputunit3,", "SSEoutputunit4,", "SSEoutputunit5,", "SSEoutputunit6,", "SSEoutputunit7,", "SSEoutputunit8,"])

        # Writes the hidden unit encoding values for each of the hidden units to a csv file. 
        row = ["HiddenUnit1Encoding,", "HiddenUnit2Encoding,", "HiddenUnit3Encoding,"]
        HUE_1 = open("HiddenUnitEncoding_10000000.csv", 'w+', newline='')
        write_HUE_1 = csv.writer(HUE_1)
        write_HUE_1.writerow(row)
        writers.append(write_HUE_1)

        HUE_2 = open("HiddenUnitEncoding_01000000.csv", 'w+', newline='')
        write_HUE_2 = csv.writer(HUE_2)
        write_HUE_2.writerow(row)
        writers.append(write_HUE_2)

        HUE_3 = open("HiddenUnitEncoding_00100000.csv", 'w+', newline='')
        write_HUE_3 = csv.writer(HUE_3)
        write_HUE_3.writerow(row)
        writers.append(write_HUE_3)

        HUE_4 = open("HiddenUnitEncoding_00010000.csv", 'w+', newline='')
        write_HUE_4 = csv.writer(HUE_4)
        write_HUE_4.writerow(row)
        writers.append(write_HUE_4)

        HUE_5 = open("HiddenUnitEncoding_00001000.csv", 'w+', newline='')
        write_HUE_5 = csv.writer(HUE_5)
        write_HUE_5.writerow(row)
        writers.append(write_HUE_5)

        HUE_6 = open("HiddenUnitEncoding_00000100.csv", 'w+', newline='')
        write_HUE_6 = csv.writer(HUE_6)
        write_HUE_6.writerow(row)
        writers.append(write_HUE_6)

        HUE_7 = open("HiddenUnitEncoding_00000010.csv", 'w+', newline='')
        write_HUE_7 = csv.writer(HUE_7)
        write_HUE_7.writerow(row)
        writers.append(write_HUE_7)

        HUE_8 = open("HiddenUnitEncoding_00000001.csv", 'w+', newline='')
        write_HUE_8 = csv.writer(HUE_8)
        write_HUE_8.writerow(row)
        writers.append(write_HUE_8)

        # Initialize a Hidden Representations file that will be written into for task 4.
        HRF = open("HiddenRepresentationsFile.csv", 'w', newline='')
        write_HRF = csv.writer(HRF)

        k_outputs = []
        # Performs the backpropagation algorithm based on the class notes.
        while n_epochs <= epochs:   
            for d in range(len(data)):
                inputs = [1]
                inputs.extend(data[d][0])
                h_outputs, k_outputs = self.forward(inputs)
                target = data[d][1]

                out_errors = []
                h_errors = []
                error_SSE = [0] * self.n_out
                for k in range(self.n_out):
                    error = target[k] - k_outputs[k]
                    delta_k = k_outputs[k] * (1 - k_outputs[k]) * error # (/partial E_total / /partial o_k) * (/partial o_k / /partial net o_k) 
                    out_errors.append(delta_k)
                    error_SSE[k] += error**2 

                for h in range(self.n_h + 1):
                    sum = 0
                    for k in range(len(out_errors)):
                        sum += k_weights[k][h] * out_errors[k] # (/partial net o_k / /partial o_h) * (/partial E_o_k / /partial net o_k)
                    
                    delta_h = h_outputs[h] * (1 - h_outputs[h]) * sum # (/partial o_h / /partial net o_h) * (/partial E_total / /partial o_h)
                    h_errors.append(delta_h)

                # Updates weights for the second layer.
                for k in range(self.n_out):
                    for h in range(self.n_h + 1):
                        Delta_k = self.lr * out_errors[k] * h_outputs[h]
                        k_weights[k][h] = k_weights[k][h] + Delta_k

                # Updates weights for first layer.
                for h in range(1, self.n_h + 1):
                    for i in range(self.n_in + 1):
                        Delta_h = self.lr * h_errors[h] * inputs[i]
                        h_weights[h-1][i] = h_weights[h-1][i] + Delta_h
                
                h_outputs, k_outputs = self.forward(inputs)
                writers[d].writerow([numpy.round(h_outputs[i], 5) for i in range(1,self.n_h+1)])
                
                # Writes the input value, hidden values, and output value to the Hidden Representations file.
                if n_epochs == 4999:
                    temp = []
                    temp.append("Input Value: " + str(data[d][0]))
                    temp.append("Hidden Values: " + str([int(numpy.round(h_outputs[i], 0)) for i in range(1,self.n_h+1)]) ) # Cast the value as an int.
                    temp.append("Output Value: " + str(data[d][1]))
                    write_HRF.writerow(temp) 

            write_SSE.writerow([numpy.round(error_SSE[i], 5) for i in range(self.n_out)])
            n_epochs += 1 # Iterate through the number of epochs.
            epoch_counter.append(n_epochs) # Append the epoch number to be used for plotting.

        # Close each file.
        SSE_file.close()
        HUE_1.close()
        HUE_2.close()
        HUE_3.close()
        HUE_4.close()
        HUE_5.close()
        HUE_6.close()
        HUE_7.close()
        HUE_8.close()
        HRF.close()

        # Create the SSE plot and each of the HUE plots.
        self.SSE_plot(epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_10000000.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_01000000.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00100000.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00010000.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00001000.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00000100.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00000010.csv", epoch_counter)
        self.HUE_plot("HiddenUnitEncoding_00000001.csv", epoch_counter)


if __name__ == "__main__":
    # Initialize a neural network.
    nn = NN(8,3,8)

    # Initialize the set of training data.
    d = [ 
    ( [1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0] ),
    ( [0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0] ),
    ( [0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0] ),
    ( [0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0] ),
    ( [0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0] ),
    ( [0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0] ),
    ( [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0] ),
    ( [0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1] )
    ]

    # Perform the backpropagation algorithm.
    nn.backpropagate(d)
