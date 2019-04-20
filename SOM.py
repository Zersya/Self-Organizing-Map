import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#==================================================================

datas = pd.read_csv("Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv")

maxPosRandNeuron = 14
minPosRandNeuron = 8
countNeurons = 15

r = 0.1555
constTimeR = 2
o = 0.2365
constTimeO = 2

neurons = []
inputVector = []

#==================================================================

def neighboorFunction(dist, omega):
    return math.exp(-(math.pow(dist, 2) / (2 * math.pow(omega, 2))))

def updateNeighboor(t):
    return o * math.exp(-(t / constTimeO))

def updateRate(t):
    return r * math.exp(-(t / constTimeR))

def getBMU(row, it, rate, omega):
    dists = []
    x = np.array([row[0], row[1]])
    for i, neuron in enumerate(neurons, 0):
        dists.append([round(np.linalg.norm(x - neuron), 3), i])

    minDist = int(min(dists)[1])
    winner = np.array([neurons[minDist][0], neurons[minDist][1]])


    for i, neuron in enumerate(neurons, 0):
        neighboor = np.array([neuron[0], neuron[1]])
        dist = round(np.linalg.norm(winner - neighboor), 3)
        
        deltaW = rate * neighboorFunction(dist, omega) * np.subtract(x, neighboor)
        neurons[i] += deltaW

# ==================================================================

def main():
    convergen = False
    iteration = 0
    rate = r
    omega = o
    for i in range(countNeurons):
        neurons.append([round(random.uniform(minPosRandNeuron, maxPosRandNeuron), 3), round(random.uniform(minPosRandNeuron, maxPosRandNeuron), 3)])

    x, y = zip(*neurons)
    plt.scatter(x, y, s=np.pi*7, c='red', alpha=0.5, label="Before Process")
    
    plt.scatter(datas['v0'], datas['v1'], s=np.pi*7, c='grey', alpha=0.5, label="Input Vector")
    while not convergen:
        datas.apply(getBMU, axis=1, it=iteration, omega=omega, rate=rate)

        if(iteration == 100): 
            x, y = zip(*neurons)
            plt.scatter(x, y, s=np.pi*7, c='blue', alpha=0.5, label="After Process")
            convergen=True
        else:
            rate = updateRate(iteration)
            omega = updateNeighboor(iteration)
            iteration+=1


    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.show()

# ==================================================================

main()