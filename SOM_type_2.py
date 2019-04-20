import numpy
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

#==================================================================

datas = pd.read_csv("Tugas 2 ML Genap 2018-2019 Dataset Tanpa Label.csv", header=None)

inputVektor = datas.values
neurons = []

rate = 0.1555
omega = 0.12
tO = 2
tR = 2
jumlahNeurons = 15
jumlahIterasi = 100

mOmega = omega
mRate = rate

def neighborfunction(winner, neighbor, t):
    return math.exp(-(math.pow(distanceNeighbor(winner, neighbor), 2))/(2*math.pow(mOmega, 2)))

def updateNeighbor(t):
    return omega * math.exp(-(t/tO))

def updateLearningRate(t):
    return rate * math.exp(-(t/tR))

def distanceNeighbor(winner, neighbor):
    winner = numpy.array([winner[0], winner[1]])
    neighbor = numpy.array([neighbor[0], neighbor[1]])

    return numpy.linalg.norm(winner - neighbor)


for i in range(jumlahNeurons):
    neurons.append([random.uniform(7, 14), random.uniform(7, 14)])

x, y = zip(*inputVektor)
plt.scatter(x, y, s=20, c="grey", alpha=0.5)
x, y = zip(*neurons)
plt.scatter(x, y, s=20, c="green", alpha=0.5)

for t in range(jumlahIterasi):
    for x in inputVektor:
        allDistance = []
        for i, j in enumerate(neurons, 0):
            distanceX = numpy.linalg.norm(j - x)
            allDistance.append([distanceX, i])
        winner = min(allDistance)[1]
        for i, j in enumerate(neurons, 0):
            deltaW = mRate * neighborfunction(neurons[winner], j, t) * numpy.subtract(x, j)
            neurons[i] = neurons[i] + deltaW

    mOmega = updateNeighbor(t)
    mRate = updateLearningRate(t)


x, y = zip(*neurons)
plt.scatter(x, y, s=20, c="blue", alpha=0.5)
    
plt.show()