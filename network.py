
import csv
import numpy
import math
from random import random

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

        self.lr = lr
        self.n_in = inputs
        self.n_h = hidden
        self.n_out = outputs
        self.weights = []   
        
        if weights is None:
          self.weights = [[random()*0.2 - 0.1 for i in range(self.n_in + 1)] for j in range(self.n_h)]
          self.weights.extend([[random()*0.2 - 0.1 for i in range(self.n_h + 1)] for j in range(self.n_out)])
        else:
            self.weights = weights
 

    def sigmoid(self, dotprod):
        return float( 1 / ( 1 + ( math.e**(-1*dotprod) ) ) )


    def forward(self, data):
        """
	    Performs forward propagation in network 
	    Arguments
	    --------------------------------------------------------------------
		    data : list : training instance containing inputs
            EXAMPLE: data = { ([1,2,3], [1,2,3]) , ([1,2,3], [1,2,3]) , ([1,2,3], [1,2,3]) }
        Returns:      
	    --------------------------------------------------------------------
            output_h : list : list of outputs from hidden units 
            output_k : list : list of outputs from output units
	    """

       #initializing hidden units & weights, and output units & weights
        h_output = [1,0,0,0]                  # result of hidden units after utilizing input units
        h_weights = self.weights[:self.n_h]      # first weight is for constant, 8 after are for inputs
        
        
        k_output = [0,0,0,0,0,0,0,0]        # result of output units after utilizing hidden units
        k_weights = self.weights[self.n_h:]      # first weight is for constant, 8 after are for inputs 

        for h in range(1, len(h_output)):                                # taking input units into hidden units
            dotprod = numpy.dot(h_weights[h-1], data)
            h_output[h] = self.sigmoid(dotprod)

        for k in range(len(k_output)):                                        # taking hidden units as inputs for output units
            dotprod = numpy.dot(k_weights[k], h_output)
            k_output[k] = self.sigmoid(dotprod)
        return h_output, k_output
        

    def back_propagate(self,data,epochs=5000):  
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

        h_weights = self.weights[:self.n_h]
        k_weights = self.weights[self.n_h:]

        writers = []

        SSE_file = open("SumOfSquaredErrors.csv", 'w', newline='')
        write_SSE = csv.writer(SSE_file)
        write_SSE.writerow(["SSEoutputunit1,", "SSEoutputunit2,", "SSEoutputunit3,", "SSEoutputunit4,", "SSEoutputunit5,", "SSEoutputunit6,", "SSEoutputunit7,", "SSEoutputunit8,"])

        row = ["HiddenUnit1Encoding,", "HiddenUnit2Encoding,", "HiddenUnit3Encoding,"]
        HUE_1 = open("HiddenUnitEncoding_10000000.csv", 'w', newline='')
        write_HUE_1 = csv.writer(HUE_1)
        write_HUE_1.writerow(row)
        writers.append(write_HUE_1)

        HUE_2 = open("HiddenUnitEncoding_01000000.csv", 'w', newline='')
        write_HUE_2 = csv.writer(HUE_2)
        write_HUE_2.writerow(row)
        writers.append(write_HUE_2)

        HUE_3 = open("HiddenUnitEncoding_00100000.csv", 'w', newline='')
        write_HUE_3 = csv.writer(HUE_3)
        write_HUE_3.writerow(row)
        writers.append(write_HUE_3)

        HUE_4 = open("HiddenUnitEncoding_00010000.csv", 'w', newline='')
        write_HUE_4 = csv.writer(HUE_4)
        write_HUE_4.writerow(row)
        writers.append(write_HUE_4)

        HUE_5 = open("HiddenUnitEncoding_00001000.csv", 'w', newline='')
        write_HUE_5 = csv.writer(HUE_5)
        write_HUE_5.writerow(row)
        writers.append(write_HUE_5)

        HUE_6 = open("HiddenUnitEncoding_00000100.csv", 'w', newline='')
        write_HUE_6 = csv.writer(HUE_6)
        write_HUE_6.writerow(row)
        writers.append(write_HUE_6)

        HUE_7 = open("HiddenUnitEncoding_00000010.csv", 'w', newline='')
        write_HUE_7 = csv.writer(HUE_7)
        write_HUE_7.writerow(row)
        writers.append(write_HUE_7)

        HUE_8 = open("HiddenUnitEncoding_00000001.csv", 'w', newline='')
        write_HUE_8 = csv.writer(HUE_8)
        write_HUE_8.writerow(row)
        writers.append(write_HUE_8)
        k_outputs = []
        while n_epochs <= epochs:   
            for d in range(len(data)):
                inputs = [1]
                inputs.extend(data[d][0])
                h_outputs, k_outputs = self.forward(inputs)
                target = data[d][1]

                #print ("output units: ", k_outputs )
                #print ("target values: ", target)
                out_errors = []
                h_errors = []
                error_SSE = [0] * self.n_out
                for k in range(self.n_out):
                    error = target[k] - k_outputs[k]
                    delta_k = k_outputs[k] * (1 - k_outputs[k]) * error # (/partial E_total / /partial o_k) * (/partial o_k / /partial net o_k) 
                    out_errors.append(delta_k)
                    error_SSE[k] += error**2 
                #print("output errors: ", out_errors)

                for h in range(self.n_h + 1):
                    sum = 0
                    for k in range(len(out_errors)):
                        sum += k_weights[k][h] * out_errors[k] # (/partial net o_k / /partial o_h) * (/partial E_o_k / /partial net o_k)
                   # print("sum",sum,"\n") 
                    delta_h = h_outputs[h] * (1 - h_outputs[h]) * sum # (/partial o_h / /partial net o_h) * (/partial E_total / /partial o_h)
                    h_errors.append(delta_h)
                #print("hidden errors: ", h_errors)

                #updating weights for second layer
                for k in range(self.n_out):
                    for h in range(self.n_h + 1):
                        Delta_k = self.lr * out_errors[k] * h_outputs[h]
                        k_weights[k][h] = k_weights[k][h] + Delta_k

                #updating weights for first layer
                for h in range(self.n_h):
                    for i in range(self.n_in + 1):
                        Delta_h = self.lr * h_errors[h] * inputs[i]
                        h_weights[h][i] = h_weights[h][i] + Delta_h
                
                h_outputs, k_outputs = self.forward(inputs)
                writers[d].writerow([numpy.round(h_outputs[i], 8) for i in range(1,self.n_h+1)])
            write_SSE.writerow([numpy.round(error_SSE[i], 8) for i in range(self.n_out)])
            n_epochs += 1
        
            


if __name__ == "__main__":
    nn = NN(8,3,8)
    d = [ 
    [ [1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0] ],
    [ [0,1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0] ],
    [ [0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0] ],
    [ [0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0] ],
    [ [0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0] ],
    [ [0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,0] ],
    [ [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,1,0] ],
    [ [0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,1] ]
    ]
    nn.back_propagate(d, 5000)
    print(nn.forward([1,0,0,0,1,0,0,0,0])[1])
