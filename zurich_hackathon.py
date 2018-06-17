from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense # for hidden layers
from sklearn.metrics import confusion_matrix

"""Load data"""
data = pd.read_excel("zurich_insurance.xlsx")
data.head() # first 5 lines
data.tail() # show only the last 5 lines

"""Preprocessing"""
# Obtaining a Complete Dataset (Dropping Missing Values)
data = data.dropna()
data = data.reset_index(drop = True)
# data.describe()

# Change column to obtain the years the customer has been with the company
current_time = datetime.now()
customer_since = data['Customer since']
i = 0
time_since = [None] * len(data.index)
for idx in data.index:
    diff_time = current_time.timestamp() - customer_since[idx].timestamp() # in seconds
    time_since[i] = round(diff_time/60/60/24/30/12) # in years
    i+=1

data['Customer since'] = time_since
data = data.rename(columns = {'Customer since': 'Years customer'})

# Split data into predictors and outcome
X = data.iloc[:, 3:20] # from age on
y = data.iloc[:, 20] # variable to be predicted

labelencoder = LabelEncoder() 
canton = labelencoder.fit_transform(data['Canton'])
onehotencoder = OneHotEncoder(categorical_features = [0])
canton = onehotencoder.fit_transform(canton.reshape(-1,1)).toarray()
canton = canton[:, 1:] # avoid falling into dummy variable trap (e.g. if its not male is female)

encoded = pd.DataFrame(canton)
X = pd.concat([X, encoded], axis=1)

labelencoder = LabelEncoder()
X['Gender'] = labelencoder.fit_transform(X['Gender'])

# encode  output class values as integers
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Let's make the ANN!
# define baseline model
def baseline_model():
	# Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = baseline_model, epochs = 10, batch_size = 10)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=kfold)
mean = accuracies.mean()
std = accuracies.std()
print("Baseline: %.2f%% (%.2f%%)" % (mean*100, std*100))

# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
classifier.fit(X_train, y_train, batch_size=10, epochs=10)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred)
#esb = confusion_matrix(y_test.argmax(axis=1), y_pred.round().argmax(axis=1))

#from sklearn.externals import joblib
## save the model to disk
#filename = 'finalized_model.pkl'
#joblib.dump(estimator, filename)
#
## some time later...
# 
## load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
#print(result)

import numpy
numpy.unique(y_pred)

