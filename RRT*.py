import numpy as np
import matplotlib.pyplot as plt
import random

np.set_printoptions(precision=3, suppress=True)


class treeNode():
    def __init__(self, locationX, locationY):
        self.locationX = locationX
        self.locationY = locationY
        self.children = []
        self.parent = None


class RRTStarAlgorithm():
    def __init__(self, start, goal, numIterations, grid, stepSize):
        self.randomTree = treeNode(start[0], start[1])
        self.goal = treeNode(goal[0], goal[1])
        self.nearestNode = None
        self.iterations = min(numIterations, 700)
        self.grid = grid
        self.rho = stepSize
        self.nearestDist = 10000
        self.numWaypoints = 0
        self.Waypoints = []
        self.searchRadius = self.rho * 2
        self.neighbouringNodes = []
        self.goalArray = np.array([goal[0], goal[1]])
        self.goalCosts = [10000]

    def addChild(self, treeNode):
        if treeNode.locationX == self.goal.locationX:
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        else:
            self.nearestNode.children.append(treeNode)
            treeNode.parent = self.nearestNode

    def sampleAPoint(self):
        x = random.randint(1, grid.shape[1])
        y = random.randint(1, grid.shape[0])
        point = np.array([x, y])
        return point

    def steerToPoint(self, locationStart, locationEnd):
        offset = self.rho * self.unitVector(locationStart, locationEnd)
        point = np.array([locationStart.locationX + offset[0], locationStart.locationY + offset[1]])
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1] - 1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0] - 1
        return point

    def isInObstacle(self, locationStart, locationEnd):
        u_hat = self.unitVector(locationStart, locationEnd)
        testPoint = np.array([0.0, 0.0])
        dist = self.distance(locationStart, locationEnd)
        for i in range(int(dist)):
            testPoint[0] = min(grid.shape[1] - 1, locationStart.locationX + i * u_hat[0])
            testPoint[1] = min(grid.shape[0] - 1, locationStart.locationY + i * u_hat[1])
            if self.grid[round(testPoint[1]), round(testPoint[0])] == 1:
                return True
        return False

    def unitVector(self, locationStart, locationEnd):
        v = np.array([locationEnd[0] - locationStart.locationX, locationEnd[1] - locationStart.locationY])
        u_hat = v / np.linalg.norm(v)
        return u_hat

    def findNearest(self, root, point):
        if not root:
            return
        dis = self.distance(root, point)
        if dis <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dis
        for child in root.children:
            self.findNearest(child, point)

    def findNeighbouringNodes(self, root, point):
        if not root:
            return
        dis = self.distance(root, point)
        if dis <= self.searchRadius:
            self.neighbouringNodes.append(root)
        for child in root.children:
            self.findNeighbouringNodes(child, point)


    def distance(self, node1, point):
        dist = np.sqrt((node1.locationX - point[0]) ** 2 + (node1.locationY - point[1]) ** 2)
        return dist

    def goalFound(self, point):
        dis = self.distance(self.goal, point)
        if dis <= self.rho:
            return True
        return False

    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000
        self.neighbouringNodes = []

    def retracePath(self):
        self.numWaypoints = 0
        self.Waypoints = []
        goalCost = 0
        goal = self.goal
        while goal.locationX != self.randomTree.locationX:
            self.numWaypoints += 1
            a = np.array([goal.locationX, goal.locationY])
            self.Waypoints.insert(0, a)
            goalCost += self.distance(goal, np.array([goal.parent.locationX, goal.parent.locationY]))
            goal = goal.parent
        self.goalCosts.append(goalCost)


    def findPathDistance(self, node):
        costFromRoot = 0
        currentNode = node
        while currentNode.locationX != self.randomTree.locationX:
            costFromRoot += self.distance(currentNode,np.array([currentNode.parent.locationX, currentNode.parent.locationY]))
            currentNode = currentNode.parent
        return costFromRoot


grid = np.load('cspace.npy')
start = np.array([300.0, 300.0])
goal = np.array([1400.0, 775.0])
numIterations = 700
stepSize = 75
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

fig = plt.figure("RRT Star Algorithm")
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
plt.xlabel('X-axis $(m)$')
plt.ylabel('Y-axis $(m)$')

rrtStar = RRTStarAlgorithm(start, goal, numIterations, grid, stepSize)
plt.pause(2)


for i in range(rrtStar.iterations):

    rrtStar.resetNearestValues()

    print("Iteration: ", i)

    n = rrtStar.sampleAPoint()
    rrtStar.findNearest(rrtStar.randomTree, n)
    new = rrtStar.steerToPoint(rrtStar.nearestNode, n)
    if not rrtStar.isInObstacle(rrtStar.nearestNode, new):
        rrtStar.findNeighbouringNodes(rrtStar.randomTree, new)
        min_cost_node = rrtStar.nearestNode
        min_cost = rrtStar.findPathDistance(min_cost_node)
        min_cost = min_cost + rrtStar.distance(rrtStar.nearestNode, new)
        for node in rrtStar.neighbouringNodes:
            cost = rrtStar.findPathDistance(node)
            cost += rrtStar.distance(node, new)
            if rrtStar.isInObstacle(node, new) and cost < min_cost:
                min_cost = cost
                min_cost_node = node
        rrtStar.nearestNode = min_cost_node
        newNode = treeNode(new[0], new[1])
        rrtStar.addChild(newNode)
        plt.pause(0.01)
        plt.plot([rrtStar.nearestNode.locationX, new[0]], [rrtStar.nearestNode.locationY, new[1]], 'go', linestyle="--")

        for node in rrtStar.neighbouringNodes:
            ncost = min_cost
            ncost += rrtStar.distance(node, new)
            d = rrtStar.findPathDistance(node)
            if rrtStar.isInObstacle(node, new) and ncost < d:
                node.parent = newNode

        point = np.array([newNode.locationX, newNode.locationY])
        if rrtStar.goalFound(point):
            projectedCost = rrtStar.findPathDistance(newNode) + rrtStar.distance(rrtStar.goal, point)
            if projectedCost < rrtStar.goalCosts[-1]:
                rrtStar.addChild(rrtStar.goal)
                plt.plot([rrtStar.nearestNode.locationX, rrtStar.goalArray[0]],
                         [rrtStar.nearestNode.locationY, rrtStar.goalArray[1]], 'go', linestyle="--")
                rrtStar.retracePath()
                print("Goal Cost: ", rrtStar.goalCosts)
                plt.pause(0.25)
                rrtStar.Waypoints.insert(0, start)
                for i in range(len(rrtStar.Waypoints) - 1):
                    plt.plot([rrtStar.Waypoints[i][0], rrtStar.Waypoints[i + 1][0]],
                             [rrtStar.Waypoints[i][1], rrtStar.Waypoints[i + 1][1]], 'ro', linestyle="-")
                    plt.pause(0.01)
plt.show()

print("Goal Costs: ", rrtStar.goalCosts[1:-1])
