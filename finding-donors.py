#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:07:00 2019

@author: ichraf
"""

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# Import supplementary visualization code visuals.py
import visuals as vs

# Load the Census dataset
data = pd.read_csv("census.csv")

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (float(n_greater_50k)/n_records)*100

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])


# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()

# Initialize features dataframe with the numerical features
features = features_raw[numerical]
# Get name of categorical features
categorical_features = [x for x in features_raw if features_raw[x].dtype == object]
# Get dummies for each categorical feature and concatenate it to the features dataframe
for feature in categorical_features:
    features = pd.concat([features, pd.get_dummies(features_raw[feature], prefix = feature)], axis=1)

# TODO: Encode the 'income_raw' data to numerical values
income = pd.get_dummies(income_raw)['>50K']

# Import train_test_split
from sklearn.cross_validation import train_test_split
# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# TODO: Calculate accuracy
# Naive predictions
predictions = np.ones(n_records)
accuracy = np.mean(predictions == income)

# TODO: Calculate F-score using the formula above for beta = 0.5
beta = 0.5 # change here the importance of precision
precision = float(np.sum(predictions == income))/n_records
# Recall is 1 as all the positive samples have been identified
fscore = (1+beta**2)*precision/((beta**2*precision)+1)

# Print the results 
print ("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

from sklearn.decomposition import PCA

features_reduced = PCA(n_components=2).fit_transform(features)


# Plot negative samples
mask_neg = np.array(income==0, dtype=bool)
plt.scatter(features_reduced[mask_neg, 0], features_reduced[mask_neg, 1], color = 'pink',
            marker = 'o', edgecolors = 'black')


mask_pos = np.array(income==1, dtype=bool)
plt.scatter(features_reduced[mask_pos,0], features_reduced[mask_pos,1], color = 'green',
            marker = '^', edgecolors = 'black')
plt.legend(['<=50K','>50K'])

plt.show()





# Move to 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

features_reduced = PCA(n_components=3).fit_transform(features)

# Plot negative samples
mask_neg = np.array(income==0, dtype=bool)
ax.scatter(features_reduced[mask_neg, 0], features_reduced[mask_neg, 1], features_reduced[mask_neg, 2],
            color = 'pink', marker = 'o', edgecolors = 'black')

# Plot positive samples
mask_pos = np.array(income==1, dtype=bool)
ax.scatter(features_reduced[mask_pos,0], features_reduced[mask_pos,1], features_reduced[mask_pos,2],
           color = 'green', marker = '^', edgecolors = 'black')

plt.show()


# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end-start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print ("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

# TODO: Import the three supervised learning models from sklearn
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# TODO: Initialize the three models
clf_A = SVC(random_state=1990)
clf_B = GradientBoostingClassifier(random_state=1990)
clf_C = KNeighborsClassifier()
clf_D = RandomForestClassifier(n_estimators=100 , random_state=0)
clf_E = SVC(random_state=1990 ,kernel='poly')
# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(0.01*X_train.shape[0])
samples_10 = int(0.1*X_train.shape[0])
samples_100 = X_train.shape[0]

# Collect results on the learners
results = {}
j=0
for clf in [clf_A, clf_B, clf_C,clf_D,clf_E]:
    clf_name = clf.__class__.__name__+str(j)
    j +=1
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

#for i in results:
sigmoidsvc=SVC(random_state=1990 ,kernel='sigmoid')
res=train_predict(sigmoidsvc, samples, X_train, y_train, X_test, y_test)
r={}  
i=0  
for clf in [clf_A, clf_B, clf_C,clf_D,clf_E]:
    clf_name = clf.__class__.__name__+str(i)
    r[clf_name]={'f_test' : results[clf_name][2]['f_test'],'acc_test' :results[clf_name][2]['acc_test']}
    i+=1
r={}  
i=0  
for clf in ['a', 'b','c','d']:
    clf_name = clf+str(i)
    i=i+1
    
    
plt.hist()

# How much faster is GBC compared to SVM and KNN?

# GBC vs. SVM on the entire dataset
gbc_svc_trainig = (results['SVC'][2]['train_time']/results[
                    'GradientBoostingClassifier'][2]['train_time'])
gbc_svc_test = (results['SVC'][2]['pred_time']/results[
                    'GradientBoostingClassifier'][2]['pred_time'])
# GBC vs. KNN on the entire dataset
gbc_knn_trainig = (results['GradientBoostingClassifier'][2]['train_time']/
                   results['KNeighborsClassifier'][2]['train_time'])
gbc_knn_test = (results['KNeighborsClassifier'][2]['pred_time']/
                results['GradientBoostingClassifier'][2]['pred_time'])
                   
print ('GBC is {:.1f} times faster to train than SVM, and {:.1f} times slower than KNN'.format(gbc_svc_trainig, 
                                                                                              gbc_knn_trainig))


print ('GBC is {:.1f} times faster to test than SVM, and {:.1f} times faster than KNN'.format(gbc_svc_test, 
                                                                                            gbc_knn_test))

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

# TODO: Initialize the classifier
clf = GradientBoostingClassifier(random_state=1990)

# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators':[100, 300, 500], 'min_samples_split':[2, 10, 50]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=scorer, n_jobs=-1)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print ("Unoptimized model\n------")
print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))



# Visualize the relationship between each feature and the target
# to see which features have the strongest patterns

# Import useful libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Change the default seaborn style
sns.set_style("white")

# Normalize numerical features
data[numerical] = scaler.fit_transform(data[numerical])

# Label encoder for categorical variables to get clean looking plots
for feature in categorical_features:
    data[feature] = LabelEncoder().fit_transform(data[feature])

# Select columns for data visualization
viz_columns = [x for x in data.columns if (x != 'income') & (x != 'native-country')]

# Initialize subplots    
f, axarr = plt.subplots(4, 3, figsize=(15, 15)) 
f.subplots_adjust(wspace=.35)
i = 0 # row index
j = 0 # col index

# Plot histograms for numerical variables and countplots for the 
# categorical ones
for column in viz_columns:
    if column in categorical_features:
        g = sns.FacetGrid(data, hue='income', size = 5)
        g.map(sns.countplot, column, ax=axarr[i,j])
        plt.close(g.fig)
    else:
        g = sns.FacetGrid(data, hue='income', size = 5)
        g.map(sns.distplot, column, bins=np.arange(0,1,0.05),
              kde=False, ax=axarr[i,j], hist_kws = {'alpha':1})   
        plt.close(g.fig)
    if j==2:
        i = i+1
        j = 0
    else:
        j += 1

# TODO: Import a supervised learning model that has 'feature_importances_'
clf = GradientBoostingClassifier(random_state=1990)

# TODO: Train the supervised model on the training set 
model = clf.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print ("Final Model trained on full data\n------")
print ("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print ("\nFinal Model trained on reduced data\n------")
print ("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))



###############    My tests and 5labiz #####################
