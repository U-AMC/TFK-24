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
        self.heuristic = self._heuristic_function()
        self.value = self.route + self.heuristic

    def __lt__(self, other):
        if self.value == other.value:
            return self.pos < other.pos  # Tie-breaking by position
        return self.value < other.value

    def _heuristic_function(self):
        a, b, c = self.pos[0] - self.goal[0], self.pos[1] - self.goal[1], self.pos[2] - self.goal[2]
        return abs(a) + abs(b) + abs(c)  # Manhattan distance

class AStar:
    def __init__(self, grid, safety_distance, timeout, straighten=False):
        self.grid = grid
        self.safety_distance = safety_distance
        self.neighborhood = [p for p in itertools.product([0, 1, -1], repeat=3) if not (p[0] == 0 and p[1] == 0 and p[2] == 0)]
        self.straighten = straighten
        self.timeout = timeout

    def dist(self, first, second):
        a, b, c = first[0] - second[0], first[1] - second[1], first[2] - second[2]
        return sqrt(a**2 + b**2 + c**2)

    def halve_and_test(self, path):
        pt1, pt2 = path[0], path[-1]
        if len(path) <= 2:
            return path
        if self.grid.obstacleBetween(pt1, pt2):
            mid = len(path) // 2
            seg1 = self.halve_and_test(path[:mid + 1])
            seg2 = self.halve_and_test(path[mid:])
            return seg1[:-1] + seg2
        else:
            return [pt1, pt2]

    def generatePath(self, m_start, m_goal):
        # print(f"[INFO] A*: Searching for path from {m_start} to {m_goal}.")
        start, goal = self.grid.metricToIndex(m_start), self.grid.metricToIndex(m_goal)
        node = self.search_path(start, goal)
        if node is None:
            print("[ERROR] A* did not find any path!")
            return None, None

        path = []
        while node.parent:
            path.append(node.pos)
            node = node.parent
        path.append(start)
        if self.straighten:
            path = self.halve_and_test(path)
        path.reverse()

        path_m = [self.grid.indexToMetric(node) for node in path]
        distance = sum(self.dist(path_m[i - 1], path_m[i]) for i in range(1, len(path_m)))
        path_m[0] = (*path_m[0][:3], m_start[3])
        
        return path_m, distance

    def generateMinimumJerkPath(self, m_start, m_goal, num_points=200):
        path_m, distance = self.generatePath(m_start, m_goal)
        if path_m is None:
            return None, None

        # Time vector for parameterization
        t = np.linspace(0, 1, len(path_m))
        
        # Create minimum jerk trajectory for each dimension
        def minimum_jerk_trajectory(points):
            t_points = np.linspace(0, 1, len(points))
            A = np.vstack([t_points**i for i in range(6)]).T
            coeffs = np.linalg.lstsq(A, points, rcond=None)[0]
            t_fine = np.linspace(0, 1, num_points)
            return sum(c * t_fine**i for i, c in enumerate(coeffs))

        # Extract x, y, z coordinates from the path
        x = [p[0] for p in path_m]
        y = [p[1] for p in path_m]
        z = [p[2] for p in path_m]

        # Generate minimum jerk trajectories for each axis
        x_smooth = minimum_jerk_trajectory(x)
        y_smooth = minimum_jerk_trajectory(y)
        z_smooth = minimum_jerk_trajectory(z)

        # Combine the smoothed coordinates back into a list of waypoints
        smooth_path = [(x_smooth[i], y_smooth[i], z_smooth[i]) for i in range(num_points)]

        # Calculate the distance of the smooth path
        smooth_distance = sum(self.dist(smooth_path[i - 1], smooth_path[i]) for i in range(1, len(smooth_path)))

        # Add the orientation information back to the start and end points
        smooth_path[0] = (*smooth_path[0], m_start[3])
        smooth_path[-1] = (*smooth_path[-1], m_goal[3])

        return smooth_path, smooth_distance

    def search_path(self, start, goal):
        start_node = Node(start, goal=goal)
        open_queue = []
        heapq.heappush(open_queue, start_node)
        closed_set = {}

        start_time = time.time()
        while open_queue:
            if time.time() - start_time > self.timeout:
                print(f"[ERROR] A*: Timeout limit exceeded ({time.time() - start_time:.1f} s > {self.timeout:.1f} s). Ending.")
                return None

            best_node = heapq.heappop(open_queue)
            if best_node.pos in closed_set and closed_set[best_node.pos] <= best_node.value:
                continue
            closed_set[best_node.pos] = best_node.value

            if best_node.heuristic == 0:
                return best_node

            for neighbor in self.neighbors(best_node.pos):
                neighbor_node = Node(neighbor, best_node.route + self.dist(best_node.pos, neighbor), best_node, goal)
                if neighbor in closed_set and closed_set[neighbor] <= neighbor_node.value:
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
