import numpy as np
import pandas as pd


def predict(features, parameters):
    z = np.matmul(features,parameters)
    predictions = 1/(1 + np.exp(-z))
    return predictions


def compute_cost(features, parameters, target): 
    observations = len(features)
    predictions = predict(features, parameters)
    cost_positive_class = -target * np.log(predictions) 
    cost_negative_class = (1 - target) * np.log(1-predictions)
    cost = cost_positive_class - cost_negative_class
    cost = cost.sum() / observations
    return cost


def update_parameters(features, parameters, target, learning_rate):
    observations = len(features)
    predictions = predict(features,parameters)
    slopes = np.matmul(features.T, predictions - target)
    slopes /= observations
    slopes *= learning_rate
    parameters -= slopes  
    return parameters


def gradient_descent(features, parameters, target, learning_rate, iterations, verbose=True):
    log = []
    for i in range(iterations):
        cost = compute_cost(features, parameters, target)
        parameters = update_parameters(features, parameters, target, learning_rate)
        log.append(cost)
        if i % 1000 == 0 and verbose==True:
            print('Iteration: ' + str(i) + ' Cost = ' + str(cost))
    return parameters, log 


def preprocess(data):  
    # Fill missing data with mean and mode
    data.Age = data.Age.fillna(data.Age.mean())  
    data.Embarked = data.Embarked.fillna('S') 
    
    # Convert categorical features & drop unused features
    data = pd.get_dummies(data, columns = ['Sex', 'Embarked']) 
    data.drop(columns=['SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Name', 'PassengerId'], inplace=True) 
    
    # Drop target & record feature names
    features = data.drop(columns=['Survived'])
    names = features.columns 
    names = names.insert(0,'Bias Term')
    
    # Convert to numpy
    features = features.to_numpy()  
    target = data['Survived'].to_numpy()
    
    # Add bias feature initalised as zeros
    features = np.column_stack((np.ones(len(features)), features)) 
    
    return(target, features, names)


def guage_performance(features, parameters, test, threshold=0.5):
    # Make predictions
    predictions = predict(features, parameters)
    binary_predictions = [1 if prediction >= threshold else 0 for prediction in predictions]
    binary_predictions = np.asarray(binary_predictions)
    
    # Compute model accurancy 
    accurancy = sum(test == binary_predictions)/len(test)
    
    # Breakdown of predictions
    true_positive = sum((binary_predictions == 1) & (test == 1))
    false_negative = sum((binary_predictions == 0) & (test == 0))
    false_positive = sum((binary_predictions == 1) & (test != 1))
    true_negative = sum((binary_predictions == 0) & (test != 0))

    # Compute performance metrics
    recall = true_positive/(true_positive + true_negative)
    precision = true_positive/(true_positive + false_positive)
    F_score = 2 * (precision * recall)/(precision + recall)
    
    # Results dict_
    performance = dict()
    performance['accurancy'] = accurancy
    performance['recall'] = recall
    performance['precision'] = precision
    performance['F_score'] = F_score
    
    return  performance