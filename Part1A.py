import numpy as np
from time import process_time

# test_data reads data from regression testData into 2D array
test_data = np.genfromtxt('C:/CIT/Semester 1/PML/Assignment1/data/data/regression/testData.csv',delimiter=',')

# training_data reads data from regression Trainingdata into 2D array
training_data = np.genfromtxt('C:/CIT/Semester 1/PML/Assignment1/data/data/regression/trainingData.csv',delimiter=',')

#training_data is 2d array, test_data is 1d array
def calculate_distances(training_data_feature,test_data_feature):
    euclidian_dist = np.sqrt(np.sum(np.square(training_data_feature - test_data_feature),axis =1 )) 
    return euclidian_dist

def predict(training_data_feature,test_data_feature):
    euclidian_dist = []   
    k= 3
    #calculates the distance for each all test_data on training_data   
    euclidian_dist = calculate_distances(training_data_feature,test_data_feature)
    kminimum_distance_Index = np.argsort(euclidian_dist)[:k]
    
    #inverse distance weight 
    predicted_regression_avg = np.sum(training_data[kminimum_distance_Index,-1] * (1/(euclidian_dist[kminimum_distance_Index])))/np.sum(1/(euclidian_dist[kminimum_distance_Index]))
    
    return  predicted_regression_avg

def calculate_r2(test_data,predicted_avg):
    square_residuals = np.sum(np.square(predicted_avg - test_data[:,-1]))
    sum_squares = np.sum(np.square(np.mean(test_data[:,-1]) - test_data[:,-1]))
    r2 = 1 - (square_residuals/sum_squares)
    print("Squared Error :" , r2)
    pass

#Execution point for the program    
if __name__=="__main__":
    
    predicted_avg = [] 
    for i in range(len(test_data)):
        predicted_avg.append(predict(training_data[:,:-1],test_data[i,:-1]))  
        
    calculate_r2(test_data,predicted_avg)
    print("Processed time for KNN Regression :",process_time())