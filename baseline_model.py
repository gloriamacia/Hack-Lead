
from keras.models import Sequential
from keras.layers import Dense # for hidden layers

def baseline_model():
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform',
                         activation = 'relu', input_dim = 43))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform',
                         activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform',
                         activation = 'softmax'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    
    return classifier
