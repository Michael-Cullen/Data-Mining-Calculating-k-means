import pandas as pd
import numpy as np
from matplotlib import pyplot as plot
from copy import deepcopy

plot.rcParams['figure.figsize'] = (16, 9)
plot.style.use('ggplot')

#k = 3 
maximumIterations = 100 #Limits the number of iterations of the centroid convergence

plantData = pd.read_csv("CMP3744M_ADM_Assignment 1_Task2 - dataset - plants.csv") #Data read from csv

feat1 = plantData['stem_length'] 
feat2 = plantData['stem_diameter']
feat3 = plantData['leaf_length']
feat4 = plantData['leaf_width']

f1_array = np.array(feat1.values.tolist()) #Converts features to a numpy array
f2_array = np.array(feat2.values.tolist())
f3_array = np.array(feat3.values.tolist())
f4_array = np.array(feat4.values.tolist())
#dataset = np.array(list(zip(f1_array,f2_array)))
dataset = np.asarray(plantData) #Gets dataset as a numpy array


def compute_euclidean_distance(vector1, vector2, a=1): #Calculates the distance of two vectors
    return np.linalg.norm(vector1 - vector2, axis=a) #Subracts the two vectors then calculates the euclidean norm

def compute_sse(distances,clusters,k):
    sum = 0
    for x in distances: #Iterates through all the distances from point to centroid
        sum += min(x)**2 #Takes the clustered point, squares and sums all clustered values 
    return sum
     
def initialise_centroids(f1_array,f2_array,f3_array,f4_array, k): #randomly initializes the centroids
    xCent = np.random.uniform(np.min(f1_array),np.max(f1_array),size = k) #Creates random centroid coordinates within the range of the feature
    yCent = np.random.uniform(np.min(f2_array),np.max(f2_array),size = k) #for each feature
    zCent = np.random.uniform(np.min(f3_array),np.max(f3_array),size = k)
    wCent = np.random.uniform(np.min(f4_array),np.max(f4_array),size = k)
    plot.figure()
    plot.scatter(feat1,feat2, c ='black', s=7) #Plots data points
    plot.title("Initial Scatter (k = %s)"%(k))
    plot.scatter(xCent,yCent,marker='*',s=200,c='r') #Plots the random centroids
    centroids =np.array(list(zip(xCent,yCent,zCent,wCent)), dtype=np.float32) #Creates list of centroid locations
    return centroids

def kmeans(dataset,k): #cluster data into k groups
    n = dataset.shape[0] #size of dataset
    centroids=[] 
    sse = []
    centroids = initialise_centroids(f1_array,f2_array,f3_array,f4_array,k) #gets centroids
    oldCent = np.zeros(centroids.shape) #Creates empty array to store old centroids to compare to re-calculated centroids
    clusters = np.zeros(n) #Stores closest cluster for each data point
    distances = np.zeros((n,k)) #Stores distances to each centroid 
    error = compute_euclidean_distance(centroids,oldCent,None) #Initial error value
    print("-------K = [%s]-------"%(k))
    print("Initial Error:[%s]"%(error))
    iterations = 0 #Counts number of iterations performed until error = 0 
    while error != 0.0: #Repeat until convergence (no changes in centroid location)
        iterations += 1 
        for i in range(k):    
            distances[:,i] = compute_euclidean_distance(dataset,centroids[i]) #Calculates distance from points to centroids 
        clusters = np.argmin(distances, axis=1) #Assigns point to closest centroid
        oldCent = deepcopy(centroids) #Stores copy of old centroid values
        
        for i in range(k):
            centroids[i] =np.mean(dataset[clusters==i],axis=0) #Calculates new centroid values by taking mean of assigned clusters 
        error = compute_euclidean_distance(centroids,oldCent,None) #Calculates change between last iterations centroids and the current       
        print("Error:[%s]"%(error))   
        if (iterations >= maximumIterations): #Checks how many iterations have already occured. This stops the program from halting with large k values
            return  #Exits function if too many iterations
        sse.append(compute_sse(distances,clusters, k))
    print("SSE=%s"%(sse))
    fig, ax = plot.subplots()
    ax.scatter(feat1,feat2, c=clusters, s=40) #Plots datapoints
    ax.scatter(centroids[:,0],centroids[:,1], marker='*', s=200, c='black') #Plots new centroid values with clusters coloured 
    plot.title("K-means result (k = %s)"%(k))
    
    plot.figure()
    plot.xlabel("Iterations")
    plot.ylabel("SSE")
    plot.title("SSE (k = %s)"%(k))
    plot.plot(sse)
    
    print("Iterations:[%s]"%(iterations))

    return 

kmeans(dataset,3)
kmeans(dataset,4)

