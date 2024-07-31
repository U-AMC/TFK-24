import heapq
import itertools
import time
from numpy import sqrt
import numpy as np

class Node:
    def __init__(self, pos, route=0, parent=None, goal=None):
        self.route = route
        self.pos = pos
        self.parent = parent
        if goal is not None:
            self.goal = goal
        elif parent is not None:
            self.goal = self.parent.goal
        else:
            raise Exception("Goal was not specified and the node does not have any parent!")
        self.heuristic = self.__heuristicFunction()
        self.value = self.route + self.heuristic

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __heuristicFunction(self):
        a, b, c = self.pos[0] - self.goal[0], self.pos[1] - self.goal[1], self.pos[2] - self.goal[2]
        return abs(a) + abs(b) + abs(c)  # Manhattan distance

class AStar:
    def __init__(self, grid, safety_distance, timeout, straighten=True):
        self.grid = grid
        self.safety_distance = safety_distance
        self.neighborhood = [p for p in itertools.product([0, 1, -1], repeat=3) if not (p[0] == 0 and p[1] == 0 and p[2] == 0)]
        self.straighten = straighten
        self.timeout = timeout

    def dist(self, first, second):
        a, b, c = first[0] - second[0], first[1] - second[1], first[2] - second[2]
        return sqrt(a**2 + b**2 + c**2)

    def halveAndTest(self, path):
        pt1, pt2 = path[0], path[-1]
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
        print(f"[INFO] A*: Searching for path from {m_start} to {m_goal}.")
        start, goal = self.grid.metricToIndex(m_start), self.grid.metricToIndex(m_goal)
        node = self.searchPath(start, goal)
        if node is None:
            print("[ERROR] A* did not find any path!")
            return None, None

        path = []
        while node.parent:
            path.append(node.pos)
            node = node.parent
        path.append(start)
        if self.straighten:
            path = self.halveAndTest(path)
        path.reverse()

        path_m = [self.grid.indexToMetric(node) for node in path]
        distance = sum(self.dist(path_m[i - 1], path_m[i]) for i in range(1, len(path_m)))
        path_m[0] = (*path_m[0][:3], m_start[3])
        return path_m, distance

    def searchPath(self, start, goal):
        start_node = Node(start, goal=goal)
        open_queue = []
        heapq.heappush(open_queue, start_node)
        priority_grid = np.full(self.grid.dim, np.inf)

        start_time = time.time()
        while open_queue:
            if time.time() - start_time > self.timeout:
                print(f"[ERROR] A*: Timeout limit exceeded ({time.time() - start_time:.1f} s > {self.timeout:.1f} s). Ending.")
                return None

            best_node = heapq.heappop(open_queue)
            if best_node.heuristic == 0:
                return best_node

            best_node_pos = best_node.pos
            if priority_grid[best_node_pos] <= best_node.value:
                continue
            priority_grid[best_node_pos] = best_node.value

            for neighbor in self.neighbors(best_node_pos):
                neighbor_node = Node(neighbor, best_node.route + self.dist(best_node.pos, neighbor), best_node, goal)
                if priority_grid[neighbor_node.pos] <= neighbor_node.value:
                    continue
                heapq.heappush(open_queue, neighbor_node)
        print("[ERROR] A*: open node queue is empty, could not find path!")
        return None

    def neighbors(self, pos):
        neighbors = []
        for n in self.neighborhood:
            idx = (pos[0] + n[0], pos[1] + n[1], pos[2] + n[2])
            if 0 <= idx[0] < self.grid.dim[0] and 0 <= idx[1] < self.grid.dim[1] and 0 <= idx[2] < self.grid.dim[2] and not self.grid.idxIsOccupied(idx):
                neighbors.append(idx)
        return neighbors
