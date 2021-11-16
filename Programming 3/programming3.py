# CS 445 - 002
# Spring 2020
# Programming 3 - k means
# Thomas Pollard

import numpy as np
import matplotlib.pyplot as plt
import math
import random
R = 5
K = 6
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

# Cluster classifications for each data point is stored in the data right after the x, y coordinates
C = np.zeros((myData.shape[0],1))
myData = np.append(myData, C, axis=1)

# This is a list of the solutions in the form of [((c1),(c2),...,(cK),(error))]
solutions = []

for r in range(R):
    iteration = 0
    myData[:,-1] = 0
    # Initialize random starting centroids
    # Also set lastCentroids to another random starting set so we can enter the while loop
    centroidIndices = np.random.randint(myData.shape[0], size=K)
    lastCentroidIndices = np.random.randint(myData.shape[0], size=K)
    
    centroids = np.array(myData[centroidIndices,:])
    lastCentroids = np.array(myData[lastCentroidIndices,:])

    plot()

    # Save a plot of the initial set up
    #plt.savefig("plots/k"+str(K)+"r" + str(r) + "i" + str(iteration) +".png")
    while not np.all(centroids == lastCentroids):
        iteration += 1
        lastCentroids = centroids[:].copy()

        # Classify each point to its closest cluster
        for point in myData:
            # Compute distances
            distances = np.zeros(centroids.shape[0])
            for i, k in enumerate(centroids):
                distances[i] = np.linalg.norm(point[:2]-k[:2])
            # Identify the closest cluster
            closest = distances.argmin()
            point[-1] = closest
            
        plot()
        # Save a plot of this loop iteration to see the change
        #plt.savefig("plots/k"+str(K)+"r" + str(r) + "i" + str(iteration) +".png")

        # Calculate the new centroid locations
        for i, centroid in enumerate(centroids):
            pointsInClusterIndices = np.where(myData[:,-1] == i)
            pointsInCluster = myData[pointsInClusterIndices]
            centroids[i] = np.sum(pointsInCluster, axis=0)/pointsInCluster.shape[0]


    # Compute the error and save it with respect to the centroid locations
    error = 0
    for i, centroid in enumerate(centroids):
        pointsInClusterIndices = np.where(myData[:,-1] == i)
        pointsInCluster = myData[pointsInClusterIndices]
        print("Calculating error")
        for point in pointsInCluster:
            error += np.linalg.norm(point-centroid)**2
        centroids[i] = np.sum(pointsInCluster, axis=0)/pointsInCluster.shape[0]
        
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
#plt.savefig("plots/k"+str(K)+"optimum.png")

        
plt.show()
#print(centroids)
