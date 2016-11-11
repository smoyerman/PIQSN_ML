import pandas as pd
import numpy as np
import os

""" Function to grab all files of a certain format"""
def grabAllFiles(directory = ".", fileSearch = "ACS_NSQIP_PUF"):
    filesSpecified = []
    for file in os.listdir(directory):
        if file.endswith(fileSearch):
            filesSpecified.append(file)

""" Read in all the data files """
def readData(files):
    data = pd.read_table('ACS_NSQIP_PUF_05_06_vr1.txt')
    return data


from sklearn import linear_model, decomposition, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

data = pd.read_table('ACS_NSQIP_PUF_05_06_vr1.txt')

# Create linear regression object
regr = linear_model.LinearRegression()

# Clean age
data['Age'][data['Age']=='90+'] = 90
data['Age'] = data['Age'].astype(np.float)

# Clean all Yes/No categories
data = data.replace(['Yes', 'No'], [1, 0])

# Clean other categories - categories stick with how listed in the NSQIP data table
data = data.replace(['Male', 'Female'], [0, 1])
data['RACE'] = data['RACE'].replace(['Hispanic, White', 'Hispanic, Black', 'Hispanic, Color Unknown', 'Black, Not of Hispanic Origin', 'White, Not of Hispanic Origin', 'American Indian or Alaska Native', 'Asian or Pacific Islander', 'Unknown'],[0,1,2,3,4,5,6,7])
data['INOUT'] = data['INOUT'].replace(['Outpatient','Inpatient'],[0,1])

# Clean all data
data[data == -99] = np.NaN

# create new variable - weight / height
data['W/H'] = data['WEIGHT'] / data['HEIGHT']

# Grab X data - Age, weight/height ratio, days pre-op, etc. - SHOULD REMOVE OUTPATIENT FROM THIS LIST?
X = data[['VENTILAT','HXCHF','TRANSFUS','WTLOSS','BLEEDDIS','DIALYSIS','RENAFAIL','DISCANCR','EMERGNCY','INOUT']].values
X = X.astype(np.float)
# Take only non-NaN values
noNaN = (np.sum(np.isfinite(X),axis=1) == np.shape(X)[1])
X = X[noNaN,:]

# Scale X for proper machine learning
X_scale = preprocessing.scale(X)

# Let's do some prediction - days from op to discharge
Y = data['DOptoDis'].values
Y = Y[noNaN]

# Train test split, why not?
X_train, X_test, y_train, y_test = train_test_split(X_scale, Y, test_size=0.3, random_state=0)

# Train the model using the training sets
regr.fit(X_train, y_train)

# Split into test/train

# How well did we do?
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))
