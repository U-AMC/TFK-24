import itertools, time
from numpy import sqrt, argmin
import numpy as np
from queue import PriorityQueue

class Node:

    def __init__(self, pos, route=0, parent=None, goal=None):
        self.__route  = route
        self.__pos    = pos
        self.__parent = parent
        if goal is not None:
            self.__goal = goal
        elif parent is not None:
            self.__goal = self.parent.goal
        else:
            raise Exception("Goal was not specified and the node does not have any parent!")
        self.__heuristic = self.__heuristicFunction()
        self.__value     = self.route + self.heuristic

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    @property
    def value(self):
        return self.__value

    @property
    def route(self):
        return self.__route

    @property
    def heuristic(self):
        return self.__heuristic

    @property
    def parent(self):
        return self.__parent

    @property
    def goal(self):
        return self.__goal

    @property
    def pos(self):
        return self.__pos

    def __heuristicFunction(self):
        # Using Euclidean distance for heuristic
        a = self.pos[0] - self.goal[0]
        b = self.pos[1] - self.goal[1]
        c = self.pos[2] - self.goal[2]
        return sqrt(a**2 + b**2 + c**2)

class AStar():

    def __init__(self, grid, safety_distance, timeout, straighten=True):
        self.grid            = grid
        self.safety_distance = safety_distance
        self.neighborhood    = [p for p in itertools.product([0, 1, -1], repeat=3) if not (p[0] == 0 and p[1] == 0 and p[2] == 0)]
        self.straighten      = straighten
        self.timeout         = timeout

    def dist(self, first, second):
        a = first[0] - second[0]
        b = first[1] - second[1]
        c = first[2] - second[2]
        return sqrt(a**2 + b**2 + c**2)

    def halveAndTest(self, path):
        pt1 = path[0]
        pt2 = path[-1]

        if len(path) <= 2:
            return path

        if self.grid.obstacleBetween(pt1, pt2):
            mid = len(path) // 2
            seg1 = self.halveAndTest(path[:mid + 1])
            seg2 = self.halveAndTest(path[mid:])
            return seg1[:-1] + seg2
        else:
            return [pt1, pt2]

    def generatePath(self, m_start, m_goal):

        print("[INFO] A*: Searching for path from [{:.2f}, {:.2f}, {:.2f}] to [{:.2f}, {:.2f}, {:.2f}].".format(m_start[0], m_start[1], m_start[2], m_goal[0], m_goal[1], m_goal[2]))

        start = self.grid.metricToIndex(m_start)
        goal  = self.grid.metricToIndex(m_goal)

        node = self.searchPath(start, goal)
        path = []
        path_m = []
        if node is None:
            print("[ERROR] A* did not find any path!")
            return None, None
        else:
            while node.parent:
                path.append(node.pos)
                node = node.parent
            path.append(start)
            if self.straighten:
                path = self.halveAndTest(path)
            path.reverse()
            for node in path:
                path_m.append(self.grid.indexToMetric(node))

        distance = 0.0
        for i in range(1, len(path_m)):
            distance += self.dist(path_m[i - 1], path_m[i])

        path_m[0] = (path_m[0][0], path_m[0][1], path_m[0][2], m_start[3]) 

        return path_m, distance

    def searchPath(self, start, goal):
        start_node    = Node(start, goal=goal)
        open_queue    = PriorityQueue()
        priority_grid = np.full(self.grid.dim,np.inf)

        open_queue.put(start_node)

        start_time = time.time()

        while not open_queue.empty():
            best_node = open_queue.get()

            if best_node.heuristic <= 0:
                return best_node

            best_node_pos = best_node.pos

            if priority_grid[best_node_pos] <= best_node.value:
                continue

            priority_grid[best_node_pos] = best_node.value

            for neighbor in self.neighbors(best_node_pos):
                neighbor_node = Node(neighbor, best_node.route + self.dist(best_node.pos, neighbor), best_node, goal)
                if priority_grid[neighbor_node.pos] > neighbor_node.value:
                    open_queue.put(neighbor_node)

            if time.time() - start_time > self.timeout:
                print("[ERROR] A*: Timeout limit in searchPath() exceeded ({:.1f} s > {:.1f} s). Ending.".format(time.time() - start_time, self.timeout))
                return None
            
        return None

    def neighbors(self, pos):
        neigh = []
        for n in self.neighborhood:
            idx = (pos[0] + n[0], pos[1] + n[1], pos[2] + n[2])
            if 0 <= idx[0] < self.grid.dim[0] and 0 <= idx[1] < self.grid.dim[1] and 0 <= idx[2] < self.grid.dim[2] and not self.grid.idxIsOccupied(idx):
                neigh.append(idx)
        return neigh
