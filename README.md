Adeena Ahmed, Suada Demirovic, Yash Dhayal, Jason Swick
CSC 426-01
File Name: README.md
Final Project
Description: Describes the contents of the submission.

## Contents
The deliverables include:

1. **network.py**: Implementation of a multilayerneural network learner that utilizes backpropagation (D1).
2. Data Files (D2):
    i. **HiddenUnitEncoding_00000001.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,0,0,1) for the epoch of learning that was just completed.
    ii. **HiddenUnitEncoding_00000010.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,0,1,0) for the epoch of learning that was just completed.
    iii. **HiddenUnitEncoding_00000100.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,1,0,0) for the epoch of learning that was just completed.
    iv. **HiddenUnitEncoding_00001000.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,0,0,1,0,0,0) for the epoch of learning that was just completed.
    v. **HiddenUnitEncoding_00010000.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,0,1,0,0,0,0) for the epoch of learning that was just completed.
    vi. **HiddenUnitEncoding_00100000.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,0,1,0,0,0,0,0) for the epoch of learning that was just completed.
    vii. **HiddenUnitEncoding_01000000.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (0,1,0,0,0,0,0,0) for the epoch of learning that was just completed.
    viii. **HiddenUnitEncoding_10000000.csv**: A csv file containing the three hidden units in the network for the input \vec{x} = (1,0,0,0,0,0,0,0) for the epoch of learning that was just completed.
    ix. **SumOfSquaredErrors.csv**: A csv file containing the value of the sum of squared errors for each of the output units over the eight training examples for the epoch of learning that was just completed.
3. Plots (D3):
    i. **HiddenUnit_00000001_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,0,0,1) vs. the epoch number (T3.2).
    ii. **HiddenUnit_00000010_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,0,1,0) vs. the epoch number (T3.2).
    iii. **HiddenUnit_00000100_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,0,0,0,1,0,0) vs. the epoch number (T3.2).
    iv. **HiddenUnit_00001000_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,0,0,1,0,0,0) vs. the epoch number (T3.2).
    v. **HiddenUnit_00010000_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,0,1,0,0,0,0) vs. the epoch number (T3.2).
    vi. **HiddenUnit_00100000_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,0,1,0,0,0,0,0) vs. the epoch number (T3.2).
    vii. **HiddenUnit_01000000_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (0,1,0,0,0,0,0,0) vs. the epoch number (T3.2).
    viii. **HiddenUnit_10000000_Plot.png**: A graph of the values emitted by the three hidden units in the network for the input \vec{x} = (1,0,0,0,0,0,0,0) vs. the epoch number (T3.2).
    ix. **SquaredErrorsPlot.png**: A graph of the sum of squared errors vs. the epoch number (T3.1). 
4. **HiddenRepresentationsFile.csv**: Holds information corresponding to the three hidden unit values in the eight Hidden Unit Files produced (D4).
5. **D4_Analysis.pdf**: A written report of the observations and analysis regarding the hidden value encodings learned by the machine (D4).
6. **D5_Reflection.pdf**: A writeup reflecting on the assignment (D5).


## Build and Command-Line Execution Instructions for the HPC:

1) On OnDemand, go to your File Home Directory.
2) Upload Neural-Networks.tar.gz to your File Home Directory.
3) In the terminal, type in "module add python". Press the return key.
4) In the terminal, type in "pip install matplotlib". Press the return key.
5) Run the program by typing in "python3 network.py". Press the return key.