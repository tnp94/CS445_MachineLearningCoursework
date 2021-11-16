# CS 445 - 002
# Spring 2020
# Programming 3 - fuzzy c means
# Thomas Pollard

import numpy as np
import matplotlib.pyplot as plt
import math
import random
R = 5
K = 6
M = 5
plt.ion()

def plot():
    plt.clf()
    plt.show()
    for i, centroid in enumerate(centroids):
        rColor = random.random()
        gColor = random.random()
        bColor = random.random()
        randColor = (rColor, gColor, bColor)
        pointsInClusterIndices = np.where(myData[:,-1] == i)
        pointsInCluster = myData[pointsInClusterIndices]
        X = pointsInCluster[:,0]
        Y = pointsInCluster[:,1]
        plt.scatter(X,Y)
        
        CX = centroids[i,0]
        CY = centroids[i,1]
        plt.scatter(CX, CY, marker = "*", color = 'black')
    
# Load the data
myData = np.genfromtxt('cluster_dataset.txt')

X = myData[:,0]
Y = myData[:,1]
C = np.zeros((myData.shape[0],1))
myData = np.append(myData, C, axis=1)
weights = np.random.rand(myData.shape[0], K)

solutions = []
for r in range(R):
    started = False
    iteration = 0
    myData[:,-1] = 0
    # Initialize random starting centroids
    weights = np.random.rand(myData.shape[0], K)
    lastClustering = myData[:,-1].copy()
    lastCentroids = np.random.rand(K,2)
    
    
    centroids = np.zeros((K,2))
    # Compute the inital centroid location for each cluster
    for c, centroid in enumerate(centroids):
        numerator = np.zeros((1,2))
        denominator = np.zeros((1,2))
        for w, weight in enumerate(weights[:,c]):
            numerator += weight**M * myData[w,:2]
            denominator += weight**M
        centroids[c] = numerator/denominator
        print("Centroid " + str(c) + " at " + str(centroid))

    plot()
    
    # Save a plot of the initial set up
    #plt.savefig("fuzzyplots/k"+str(K)+"r" + str(r) + "i" + str(iteration) +".png")
    while not np.all(lastClustering == myData[:,-1]) or not started:
        started = True
        iteration += 1
        lastCentroids = centroids[:].copy()
        lastClustering = myData[:,-1].copy()
            
        # Classify each point to its closest cluster
        for p, point in enumerate(myData):
            # Identify the closest cluster
            closest = weights[p,:].argmax()
            point[-1] = closest
        plot()
        # Save a plot of this loop iteration to see the change
        #plt.savefig("fuzzyplots/k"+str(K)+"r" + str(r) + "i" + str(iteration) +".png")

        # For each data point, compute its coefficients/membership grades for being in the clusters (e-step)
        for p, point in enumerate(myData[:,:2]):
            for w, weight in enumerate(weights[p,:]):
                denominator = 0
                for c, centroid in enumerate(centroids):
                    denominator += ((np.linalg.norm(point-centroids[w,:])/np.linalg.norm(point-centroid))**(2/(M-1)))
                weights[p,w] = 1/denominator

        # Compute the centroid for each cluster (m-step)
        for c, centroid in enumerate(centroids):
            numerator = np.zeros((1,2))
            denominator = np.zeros((1,2))
            for w, weight in enumerate(weights[:,c]):
                numerator += (weight**M) * myData[w,:2]
                denominator += weight**M
            centroids[c] = numerator/denominator
            print("Centroid " + str(c) + " at " + str(centroid))

    # Compute the error and save it with respect to the centroid locations
    error = 0
    for i, centroid in enumerate(centroids):
        pointsInClusterIndices = np.where(myData[:,-1] == i)
        pointsInCluster = myData[pointsInClusterIndices]
        print("Calculating error")
        for point in pointsInCluster[:,:-1]:
            error += np.linalg.norm(point-centroid)**2
        centroids[i] = np.sum(pointsInCluster[:,:-1], axis=0)/pointsInCluster.shape[0]
        
    print("Error for r"+str(r)+": "+str(error))
    solutions.append((centroids, error))



    plot()    
    if r != R-1:
        plt.clf()

best = min(solutions, key = lambda x : x[-1])
centroids = best[0]
print("The centroid config with the smallest error was " + str(centroids) + " with error of " + str(best[1]))


# In the best solution found, classify each point to its closest cluster
for point in myData:
    # Compute distances
    distances = np.zeros(centroids.shape[0])
    for i, k in enumerate(centroids):
        distances[i] = np.linalg.norm(point[:2]-k[:2])
    # Identify the closest cluster
    closest = distances.argmin()
    point[-1] = closest
    
plot()
plt.figtext(0.5, 0.2,"Error: " + str(best[1]))
# Save a plot of the optimum configuration
#plt.savefig("fuzzyplots/k"+str(K)+"optimum.png")

        
plt.show()
#print(centroids)
