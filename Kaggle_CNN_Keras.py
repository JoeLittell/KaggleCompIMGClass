# -*- coding: utf-8 -*-
"""
Solar Panel Classification Team Project (Kaggle)

Duke University
Master's in Interdisiplinary Data Science
IDS 705 - Principles of Machine Learning
Instructor: Dr. Kyle Bradberry

Team Members:
    Joe Littell
    Emma Sun
    Chang Shu
    Julia Oblasova
"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
import cv2

plt.close()

'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images  = './data/training/'
dir_test_images   = './data/testing/'
dir_train_labels  = './data/labels_training.csv'
dir_test_ids      = './data/sample_submission.csv'

'''
Include the functions used for loading, preprocessing, features extraction, 
classification, and performance evaluation
'''

def load_data(dir_data, dir_labels, training = True):
    ''' Load each of the image files into memory 
    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory
    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids       = labels_pd.id.values
    data      = []
    for identifier in ids:
        fname     = dir_data + identifier.astype(str) + '.tif'
        image     = cv2.imread(fname)
        image     = image/225                    # rescale the image to make the pixels a value between 0 and 1
        #image     = cv2.resize(image, (0,0),     # Determine image to preprocess
         #                      fx=1, fy=1,       # Used to resize by factor but we keep it 101,101 pixels
          #                     interpolation = cv2.INTER_AREA) # Antialiasing
        data.append(image)
    data = np.array(data) # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids


def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    clf = Sequential()
    
    # Add a 1st conv. layer
    clf.add(Conv2D(32, (3, 3), 
                   input_shape = (101, 101, 3), 
                   activation = 'relu'))  #
    clf.add(MaxPooling2D(pool_size = (2, 2)))

    # Add a 2nd conv. layer
    clf.add(Conv2D(32, (3, 3), 
                   activation = 'relu')) # no need to specify the input shape
    clf.add(MaxPooling2D(pool_size = (2, 2)))

    # Add a 3nd conv. layer
    clf.add(Conv2D(64, (3, 3), 
                   activation = 'relu')) # no need to specify the input shape
    clf.add(MaxPooling2D(pool_size = (2, 2)))

    # Flatten the array to a 1D array
    clf.add(Flatten())

    # Full connection
    clf.add(Dense(units = 64, activation = 'relu'))
    clf.add(Dropout(0.2)) 
    clf.add(Dense(units = 1, activation = 'sigmoid'))

    # Print a summary to see number of features
    clf.summary()

    # Compile the Model
    clf.compile(loss = 'binary_crossentropy', 
            optimizer = 'rmsprop', 
            metrics = ['accuracy'])
    
    return clf

def cv_performance_assessment(X,y,k,clf):
    '''Cross validated performance assessment
    
    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use
    
    Divide the training data into k folds of training and validation data. 
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are 
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0],dtype='object')
    kf = StratifiedKFold(n_splits = k, shuffle = True)
    
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val   = X[train_index], X[val_index]
        y_train          = y[train_index]
        
        # Train the classifier        
        clf.fit(X_train, y_train, epochs = 5, batch_size = 50)
        
        # Test the classifier on the validation data for this fold      
        cpred            = clf.predict(X_val)
        
        # Save the predictions for this fold
        prediction_scores[val_index] = cpred[:,0]

    return prediction_scores

def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)
   
    plt.plot([0,1],[0,1],'--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


'''
Sample script for cross validated performance
'''

# Set parameters for the analysis
num_training_folds = 20

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)

# Choose which classifier to use
clf = set_classifier()

# Perform cross validated performance assessment
#prediction_scores = cv_performance_assessment(data,labels,num_training_folds,clf)

# Compute and plot the ROC curves
#plot_roc(labels, prediction_scores)


'''
Sample script for producing a Kaggle submission
'''

produce_submission = True # Switch this to True when you're ready to create a submission for Kaggle

if produce_submission:
    
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    clf.fit(training_data, training_labels, epochs = 5, batch_size = 50)
    
    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training = False)
    test_scores    = clf.predict(test_data)[:,0]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                   'score':  test_scores})
    submission_file.to_csv('submission4.csv',
                           columns = ['id','score'],
                           index = False)
