
import csv

class NN: 
    def __init__(self,inputs,hidden,outputs,lr = 0.3,weights = None):
        """
        Arguments
	    -------------------------------------------------------------------- 
            inputs: int : number of input units
            hidden: int : number of hidden units
            inputs: int : number of output units
            lr : float : learning rate
            weights: list: list of weights for each hidden unit
        """

        self.lr = lr
        self.n_in = inputs
        self.n_h = hidden
        self.n_out = outputs

        if weights is None:
            self.weights = [[0] * self.n_in] * self.n_h + [[0] * self.n_h] * self.n_out
        else:
            self.weights = weights
 

    def forward(self, data):
        """
	    Performs forward propagation in network 
	    Arguments
	    --------------------------------------------------------------------
		    data : list : training instance containing list of inputs

        Returns:      
	    --------------------------------------------------------------------
            output_h : list : list of outputs from hidden units 
            output_k : list : list of outputs from output units
	    """
        h_weights = self.weights[:self.n_h]
        k_weights = self.weights[self.n_h:]
    

        return [], []
        

    def back_propagate(self,data,epochs):
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
        write_SSE.writerow(row)
        writers.append(write_HUE_2)

        HUE_3 = open("HiddenUnitEncoding_00100000.csv", 'w', newline='')
        write_HUE_3 = csv.writer(HUE_3)
        write_SSE.writerow(row)
        writers.append(write_HUE_3)

        HUE_4 = open("HiddenUnitEncoding_00010000.csv", 'w', newline='')
        write_HUE_4 = csv.writer(HUE_4)
        write_SSE.writerow(row)
        writers.append(write_HUE_4)

        HUE_5 = open("HiddenUnitEncoding_00001000.csv", 'w', newline='')
        write_HUE_5 = csv.writer(HUE_5)
        write_SSE.writerow(row)
        writers.append(write_HUE_5)

        HUE_6 = open("HiddenUnitEncoding_00000100.csv", 'w', newline='')
        write_HUE_6 = csv.writer(HUE_6)
        write_SSE.writerow(row)
        writers.append(write_HUE_6)

        HUE_7 = open("HiddenUnitEncoding_00000010.csv", 'w', newline='')
        write_HUE_7 = csv.writer(HUE_7)
        write_SSE.writerow(row)
        writers.append(write_HUE_7)

        HUE_8 = open("HiddenUnitEncoding_00000001.csv", 'w', newline='')
        write_HUE_8 = csv.writer(HUE_8)
        write_SSE.writerow(row)
        writers.append(write_HUE_8)


        while n_epochs <= epochs:
            for d in range(len(data)):
                h_outputs, k_outputs = self.forward(d[0])
                inputs = data[d][0]
                target = data[d][1]

                out_errors = []
                h_errors = []
                error_SSE = [0] * self.n_out
                for k in range(self.n_out):
                    error = target[k] - k_outputs[k]
                    delta_k = k_outputs[k] * (1 - k_outputs[k]) * error # (/partial E_total//partial o_k) * (/partial o_k//partial net o_k) 
                    out_errors.append(delta_k)
                    error_SSE[k] += error**2 

                for h in range(self.n_h):
                    sum = 0
                    for k in range(len(out_errors)):
                        sum += k_weights[k][h] * out_errors[k] # (/partial net o_k//partial o_h) * (/partial E_o_k//partial net o_k)
                    
                    delta_h = h_outputs[h] * (1 - h_outputs[h]) * sum # (/partial o_h//partial net o_h) *(/partial E_total//partial o_h)
                    h_errors.append(delta_h)

                #updating weights for second layer
                for k in range(self.n_out):
                    for h in range(self.n_h):
                        Delta_k = self.lr * out_errors[k] * h_outputs[h]
                        k_weights[k][h] = k_weights[k][h] + Delta_k

                #updating weights for first layer
                for h in range(self.n_h):
                    for i in range(self.n_in):
                        Delta_h = self.lr * h_errors[h] * inputs[i]
                        h_weights[h][i] = h_weights[h][i] + Delta_h
                
                h_outputs, k_outputs = self.forward(d[0])
                writers[d].writerow([h_outputs for i in range(self.n_h)])

            write_SSE.writerow([error_SSE[i] for i in range(self.n_out)])
