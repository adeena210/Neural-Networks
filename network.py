
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
		    data : list : training instance containing list of inputs and 
            list of outputs

        Returns:      
	    --------------------------------------------------------------------
            output : list : list of outputs from hidden units and output units
	    """

        return [], []

    def error(self, data):
        pass

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

        while n_epochs <= epochs:
            for d in data:
                h_outputs, k_outputs = self.forward(d)
                inputs = d[0]
                target = d[1]


                out_errors = []
                h_errors = []
                for k in range(self.n_out):
                    delta_k = k_outputs[k] * (1 - k_outputs[k]) * (target[k] - k_outputs[k]) # (/partial E_total//partial o_k) * (/partial o_k//partial net o_k) 
                    out_errors.append(delta_k)

                for h in range(self.n_h):
                    sum = 0
                    for k in range(len(out_errors)):
                        sum += k_weights[k][h] * out_errors[k] # (/partial net o_k//partial o_h) * (/partial E_o_k//partial net o_k)
                    
                    delta_h = h_outputs[h] * (1 - h_outputs[h]) * sum # (/partial o_h//partial net o_h) *(/partial E_total//partial o_h)
                    h_errors.append(delta_h)

                for k in range(self.n_out):
                    for h in range(self.n_h):
                        Delta_k = self.lr * out_errors[k] * h_outputs[h]
                        k_weights[k][h] = k_weights[k][h] + Delta_k

                for h in range(self.n_h):
                    for i in range(self.n_in):
                        Delta_h = self.lr * h_errors[h] * inputs[i]
                        h_weights[h][i] = h_weights[h][i] + Delta_h

                        




            
 