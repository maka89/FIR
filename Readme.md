
# FIR - Real Time convolutional neural networks for time series

Library to do machine learning using Finite Impulse Response convolutions.
I.e. convolutional kernels that are large and that are skewed so that output only depends on the samples that came before.

The weights from the "FIR" type layers contained in this repo can be extracted and used as the  impulse responses in real time convolution algorithms (For instance https://github.com/HiFi-LoFi/FFTConvolver). These types of algorithms can perform convolutions in real time with impulse responses of several million samples.


## Examples
(See tests folder for more examples)
    
    from fir import Model,FIR,Activation,Dense
    
    //Generate Input and output data
    // 3 samples of length 1000. 1 Channel.
    X = np.random.randn(3,1000,1)
    Y = np.zeros_like(X)
    
    // 1-input channel, 1 output channel, kernel of length 100.
    layer1 = FIR(1,1,100)
    
    model = Model([layer1])
    model.fit(X,Y)

<!-->

     // 1-input channel, 4 output channels, kernel of length 100.
    layer1 = FIR(1,4,100)
    layer2 = Activation('tanh')
    // 4-input channel, 1 output channels, kernel of length 50.
    layer3 = FIR(4,1,50)
    
    model = Model([layer1,layer2,layer3])
    model.fit(X,Y)
    
 <!-->   
    // 1-input channel, 4 output channels, kernel of length 100.
    layer1 = FIR(1,4,100)
    layer2 = Activation('tanh')
    layer3 = Dense(4,5)
    layer4 = Activation('tanh')
    // 4-input channel, 1 output channels, kernel of length 50.
    layer5 = FIR(5,1,50)
    
    model = Model([layer1,layer2,layer3,layer4, layer5])
    model.fit(X,Y)
    
### Meta-Layers

    from fir import ParallelSum,FIR,Sequential,Split
    layer1 = FIR(1,1,100)
    
    seq1 = Sequential([FIR(1,4,50),FIR(4,1,50)])
    
    meta = ParallelSum([seq1,layer1]) //Sums results of seq1 and layer1
    model = Model([meta])
    model.fit([X1,X2],Y)
     
    #OR
    model = Model([Split(2),meta])
    model.fit(X,Y)
    
