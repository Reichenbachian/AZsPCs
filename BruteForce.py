import multiprocessing as mp
import math
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import Delaunay
import pdb
from itertools import combinations
import operator as op
import functools
import sys
import threading

'''
Constants for each problem
'''
maxamizing = True
boardSize = int(sys.argv[1])
class Node:
    '''
    tree
    parentNodes = [] #parent Nodes in list of points
    coord = None #my coordinate
    childNodes = [] #child node references
    leaf = False
    depth = None
    ----------------------------------------
    graph
    connected

    '''
    def __init__(self, v, simplicies=None, nodes=None, depth=None, parentNodes=None, rootCoord=None, parentNode=None, connected=None):
        self.coord = v
        if depth != type(None) and type(parentNodes) != type(None) and type(rootCoord) != None:
            # print("Creating tree node...")
            #Tree implementation
            self.parentNodes = parentNodes
            self.parentNode = parentNode
            self.depth = depth
            self.childNodes = []
            self.connected = None
            self.rootCoord = rootCoord
        elif type(nodes) != type(None) and type(simplicies) != type(None):
            # print("Creating graph node...")
            #Graph implementation
            self.parentNodes = None
            self.childNodes = None
            self.nodes = nodes
            nodes.append(self)
            self.connected = self.getConnected(v, simplicies, connected)
        else:
            print("Not matching either type.")
    def getConnected(self, vertex, simplicies, alreadyCreatedNodes):
        '''
        returns connected nodes to given vertex.
        '''
        connected = []
        for simplexIndex in range(len(simplicies)):
            simplex = simplicies[simplexIndex]
            vertIndex = coordInArray(simplex, vertex)
            if vertIndex != -1:#if vertex in simplex
                for checkVertIndex in range(len(simplex)):
                    alreadyConnected = False
                    for node in alreadyCreatedNodes:
                        if node.coord[0] == simplex[checkVertIndex][0] and\
                            node.coord[1] == simplex[checkVertIndex][1]:
                            alreadyConnected = True
                            #Append so bidirectional connection
                            if coordInArray(connected, node) == -1:
                                connected.append(node)
                            break
                    if vertIndex == checkVertIndex or\
                        coordInArray(connected, simplex[checkVertIndex]) != -1 or\
                        alreadyConnected:
                        continue
                    connectedArr = alreadyCreatedNodes[:]
                    connectedArr.append(self)
                    connected.append(Node(simplex[checkVertIndex],\
                        simplicies=simplicies,\
                        nodes=self.nodes, connected=connectedArr))
        return connected
    def addNode(self, point):
        self.childNodes.append(point)
    def getChildsParents(self):
        '''
        Returns child nodes parents, includes self
        '''
        retArr = self.parentNodes[:]
        retArr.append(self)
        return retArr
    def __str__(self):
        return str(self.coord)+ "->"+ str(self.connected if self.connected != None else self.childNodes)
    def __repr__(self):
        return str(self.coord)
    def __getitem__(self,key):
        if key > 2 or key < 0:
            print("INVALID KEY")
            sys.exit()
        return self.coord[int(key)]
def npr(n, r):
    def stirling(n):
        # http://en.wikipedia.org/wiki/Stirling%27s_approximation
        return math.sqrt(2*math.pi*n)*(n/math.e)**n
    return (stirling(n)/stirling(n-r) if n>20 else
            math.factorial(n)/math.factorial(n-r))
def getArea(arr):
    summation = 0
    for i in range(len(arr)):
        iAdj = (i+1)%len(arr)
        summation += arr[iAdj][0]*arr[i][1]-arr[i][0]*arr[iAdj][1]
    return abs(summation/2.0)
def coordInArray(arr, coord):
    if type(coord) is Node:
        coord = coord.coord
    '''
    Finds index of coordinate in array of coordinates or nodes
    '''
    for i in range(len(arr)):
        arrCoord=0
        if type(arr[i]) is Node:
            arrCoord = arr[i].coord
        else:
            arrCoord = arr[i]
        if len(arr) > 0 and arrCoord[0] == coord[0] and arrCoord[1] == coord[1]:
            return i
    return -1

def createSubTree(graphNode, treeNode, ofDepth = None):
    if hasEqualSlopes(treeNode.parentNodes, skipLast=True) or\
        doesIntersect(treeNode.parentNodes, skipLast=True):
        return None

    #Iterate through all of the triangles
    recordScore = 0 if maxamizing else 999999999
    bestShape = []
    myParents = treeNode.getChildsParents()
    #End statement of recursing method.
    #1) Check if depth is correct
    #2) Check if makes closed loop
    rootCoord = treeNode.rootCoord
    if len(myParents) == boardSize:
        if coordInArray(graphNode.connected, rootCoord) != -1:
            score = getArea(myParents)
            newTreeNode = Node(rootCoord, depth=treeNode.depth+1, parentNodes=myParents, rootCoord=rootCoord, parentNode=treeNode)
            treeNode.childNodes.append(newTreeNode)
            #Always return because there is only one place for it to go at this point:
            #Back to rootcoord, so no other possibilities, no reason to iterate further.
            if not doesIntersect(myParents) and not hasEqualSlopes(myParents):
                return score, myParents
        return None

    # Go through every potential path
    for node in graphNode.connected:
        coord = node.coord
        if coordInArray(myParents, node) == -1:
            newTreeNode = Node(coord, depth=treeNode.depth+1, parentNodes=treeNode.getChildsParents(), rootCoord=rootCoord, parentNode=treeNode)
            treeNode.childNodes.append(newTreeNode)
            subTreeScoreAndShape = createSubTree(node, newTreeNode, ofDepth=None)
            # Accumulate Score
            if subTreeScoreAndShape != None:
                score, shape = subTreeScoreAndShape
                # Update best score
                if (maxamizing and score > recordScore) or\
                    (not maxamizing and score < recordScore):
                    recordScore = score
                    bestShape = shape
    return recordScore, bestShape
                    
def getPolygon(arr):
    '''
    Creates a tree of possible shapes with given verticies
    and returns best one.
    '''
    # Creates a random list of valid points
    points = np.array(arr)
    try:
        #WATCH OUT. I AM DISMISSING ERROR HERE!
        tri = Delaunay(points)
    except:
        return
    triangles = points[tri.simplices]
    #Create tree
    nodes = []
    graph = Node(triangles[0][0], simplicies=triangles, nodes=nodes, connected=[])
    tree = Node(triangles[0][0], parentNodes=[], depth=0, rootCoord=triangles[0][0])
    return createSubTree(graph, tree, ofDepth=boardSize-1)
def triPlot(triangles, points):
    plt.triplot(points[:,0], points[:,1], triangles.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
def hasEqualSlopes(arr, skipLast=False):
    '''
    Checks for equal slopes between any two lines. Disallowed by the AZsPCs rules
    skipLast prevents autocompletion of last line.
    '''
    score = 0
    slopes = []
    if len(arr) <= 2:
        return False
    for i in range(len(arr)):
        if skipLast and i == len(arr)-1:
            return False
        iAdj = (i+1)%len(arr)
        if arr[i][0]-arr[iAdj][0] == 0 or arr[i][1]-arr[iAdj][1] == 0:
            return True
        slope = (arr[i][1]-arr[iAdj][1])/(arr[i][0]-arr[iAdj][0])
        if slope in slopes:
            return True
        slopes.append(slope)
    return False

def doesIntersect(arr, skipLast=False):
    '''
    Checks for intersection any two lines. Disallowed by the AZsPCs rules
    '''
    lines = []
    count = 0
    for i in range(len(arr)):
        if skipLast and i == len(arr)-1:
            break
        iAdj = (i+1)%len(arr)
        lines.append((arr[i], arr[iAdj]))
    for cLines in combinations(lines, 2):
        line1x1 = cLines[0][0][0]
        line1y1 = cLines[0][0][1]
        line1x2 = cLines[0][1][0]
        line1y2 = cLines[0][1][1]
        line2x1 = cLines[1][0][0]
        line2y1 = cLines[1][0][1]
        line2x2 = cLines[1][1][0]
        line2y2 = cLines[1][1][1]
        if line1x1 == line1x2 and line2x1 == line2x2:
            return True
        m1 = (line1y1-line1y2)/(line1x1-line1x2)
        m2 = (line2y1-line2y2)/(line2x1-line2x2)
        if m2-m1 == 0:
            return True
        b = float((m2*line2x1-line2y1-m1*line1x1+line1y1)/(m2-m1))
        a = float(-m1*(line1x1-b)+line1y1)
        minX = max(min(line1x1, line1x2), min(line2x1, line2x2))
        maxX = min(max(line1x1, line1x2), max(line2x1, line2x2))
        minY = max(min(line1y1, line1y2), min(line2y1, line2y2))
        maxY = min(max(line1y1, line1y2), max(line2y1, line2y2))
        #tol prevent floating point errors
        tol = .000000001
        if a > minX+tol and a+tol < maxX and b > minY+tol and b+tol < maxY:
            return True
    return False
def printArr(arr, add=0):
    print([(int(x[0]+add), int(x[1]+add)) for x in arr])
def uniqueRowsAndColumns(arr):
    if type(arr) is not np.array:
        arr = np.array(list(arr))
    Xs = arr[:,:1]
    ys = arr[:,1:]
    return len(Xs) == len(np.unique(Xs)) and len(ys) == len(np.unique(ys))
def plot(arr, ignore=False):
    if type(arr[0]) == Node:
        arr = [x.coord for x in arr]
    if type(arr) == None:
        if ignore == False:
            print("ARR IS NONE!")
        return
    arr = np.array(arr)
    plt.scatter(arr[:,0], arr[:,1])
    for i in range(len(arr)):
        iAdj = (i+1)%len(arr)
        plt.plot([arr[i][0],arr[iAdj][0]],
            [arr[i][1],arr[iAdj][1]])
    plt.show()

def printDepiction(arr):
    for row in range(0, len(arr)):
        for col in range(0, len(arr)):
            if coordInArray(arr, [row, col]) == -1:
                sys.stdout.write("- ")
            else:
                sys.stdout.write("* ")
        sys.stdout.write("\n")
def areThreeInRow(arr):
    for i in range(0, boardSize-2):
        iAdj1 = (i+1)%boardSize
        iAdj2 = (i+2)%boardSize
        if (arr[i][0] == arr[iAdj1][0]+1 and\
            arr[iAdj1][0] == arr[iAdj2][0]+1) or\
            (arr[i][0] == arr[iAdj1][0]-1 and\
                arr[iAdj1][0] == arr[iAdj2][0]-1):
            return True
    return False
def sortThroughReturns(result):
    global record, edgeShape, maxIt, outputCounter
    score, shape = result
    if outputCounter%outputFreq == 0:
        print(chr(27) + "[2J")
        print("Status")
        print(round(outputCounter/maxIt*100, 2), '%')
        print("Current:")
        printDepiction(shape)
        if maxamizing:
            print("Max:", end='')
        else:
            print("Min: ", end='')
        print(record, end=' ')
        printArr(edgeShape)
        printDepiction(edgeShape)
    outputCounter+=1
    # Called when asynchronous result is given.
    if (maxamizing and score > record) or\
        (not maxamizing and score < record):
        record = score
        edgeShape = shape

record = 0 if maxamizing else 9999999999999
edgeShape = []
outputCounter = 0
outputFreq = 100
maxIt = 0
def main():
    global record, edgeShape, maxIt
    pool = mp.Pool(processes=10)
    print("Generating board...")
    i = 0
    maxIt = npr(boardSize, boardSize)
    boardSeed = [x for x in range(1, boardSize+1)]
    print("Starting asynchronous Brute Force...")
    for seed in itertools.permutations(boardSeed, boardSize):
        #This statement permutes through all possible
        #x,y coordinates where the X and Y follow the
        #rules as put forth by AZsPC
        arr = [[seed[x], boardSeed[x]] for x in range(len(seed))]
        i += 1
        if areThreeInRow(arr):
            continue
        if  i > maxIt:
            break
        pool.apply_async(getPolygon, (arr,), callback=sortThroughReturns)
    pool.close()
    pool.join()

    if edgeShape != []:
        printArr(edgeShape, add=1)
        print(record)
        printDepiction(edgeShape)
        plot(edgeShape)
    else:
        print("No shapes found")


if __name__ == "__main__":
    main()