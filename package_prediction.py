from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense # for hidden layers
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np

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
X = data.iloc[:, 3:10] # from age on
y = data.iloc[:, 10:19] # variable to be predicted
labels = y.columns
y[y>1]=1



labelencoder = LabelEncoder() 
canton = labelencoder.fit_transform(data['Canton'])
onehotencoder = OneHotEncoder(categorical_features = [0])
canton = onehotencoder.fit_transform(canton.reshape(-1,1)).toarray()
canton = canton[:, 1:] # avoid falling into dummy variable trap (e.g. if its not male is female)
encoded = pd.DataFrame(canton)
X = pd.concat([X, encoded], axis=1)

# encode  class values as integers
#encoder = LabelEncoder()
#customer_lifetime_value = encoder.fit_transform(data['Customer Lifetime Value'])
# convert integers to dummy variables (i.e. one hot encoded)
#encoded_clv = np_utils.to_categorical(customer_lifetime_value)

labelencoder = LabelEncoder() 
clv = labelencoder.fit_transform(data['Customer Lifetime Value'])
onehotencoder = OneHotEncoder(categorical_features = [0])
clv = onehotencoder.fit_transform(clv.reshape(-1,1)).toarray()
clv = clv[:, 1:] # avoid falling into dummy variable trap (e.g. if its not male is female)
clv_encoded = pd.DataFrame(clv)

X = pd.concat([X, clv_encoded], axis=1)

labelencoder = LabelEncoder()
X['Gender'] = labelencoder.fit_transform(X['Gender'])




# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Let's make the ANN!
# define baseline model

classifier = Sequential()
    
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 36))

# Adding the second hidden layer
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = kfold)
#mean = accuracies.mean()
#std = accuracies.std()


# Part 3 - Making predictions and evaluating the model
# Predicting the Test set results
hist=classifier.fit(X_train, y_train, validation_split=0.25, batch_size = 10, epochs = 20)
y_pred = classifier.predict(X_test)

 
outputs = np.where(y_pred>0.5)    

pp = y_pred
pp[y_pred<0.5] = 0


results = [[] for i in range(len(y_pred))]
for i in range(len(outputs[0])):
    results[outputs[0][i]].append(labels[outputs[1][i]])

for person, i in enumerate(results):
    print(str(person) + ': ' + ', '.join(i))