# BiRRT

import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat

def dist(x_1, y_1, x_2, y_2):
    return ((x_1 - x_2)**2 + (y_1 - y_2)**2)**(0.5)

class map:
    def __init__(self):
        self.obs = [obstacle(2, 4.5, 3), 
                    obstacle(2, 3, 12),
                    obstacle(3, 15, 15)]
        self.start = (1., 1.)
        self.end = (20., 20.)
        self.max_iter = 1000
    
    def birrt(self):
        start_tree = tree(1, self.start)
        end_tree = tree(1, self.end)
        start_tree.set_goal(end_tree)
        end_tree.set_goal(start_tree)

        for counter in range(1, self.max_iter+1):
            s_new_point, s_nearest_node, s_theta = start_tree.step()
            if not self.intersects_obj(s_new_point, s_nearest_node, s_theta):
                start_tree.add_node(s_new_point, s_nearest_node)
                flag, dest_node = start_tree.check_merge()
                if flag:
                    d_dist = dist(s_new_point[0], s_new_point[1], dest_node.x, dest_node.y)
                    d_theta = ( (s_new_point[0] - dest_node.x)/d_dist , \
                               (s_new_point[1] - dest_node.y)/d_dist )
                    if not self.intersects_obj(s_new_point, dest_node, d_theta):
                        print("Converged!")
                        return self.plot(start_tree, end_tree, dest_node)
            
            start_tree, end_tree = end_tree, start_tree
            
            if counter == self.max_iter - 1:
                print("Not converged!")

    
    def plot(self, source_tree, dest_tree, dest_node):
        x, y = [], []
        for node in source_tree.nodes[1:]:
            x.append(node.x)
            y.append(node.y)
        plt.scatter(x, y, c="b", label="Source Tree")

        x, y = [], []
        for node in dest_tree.nodes[1:]:
            x.append(node.x)
            y.append(node.y)
        plt.scatter(x, y, c="r", label="Dest Tree")
        
        plt.scatter(source_tree.nodes[0].x, source_tree.nodes[0].y, c="orange", label="Start point")
        plt.scatter(dest_tree.nodes[0].x, dest_tree.nodes[0].y, c="green", label="Goal point")
        
        angles = np.linspace(0, 2*np.pi, 100)
        plt.plot(self.obs[0].c_x + self.obs[0].radius*np.cos(angles), self.obs[0].c_y + self.obs[0].radius*np.sin(angles), c = "k", label="Obstacles")
        for obs in self.obs[1:]:
            plt.plot(obs.c_x + obs.radius*np.cos(angles), obs.c_y + obs.radius*np.sin(angles), c = "k")
        
        path = [[], []]

        cur_node = source_tree.nodes[-1]
        x, y = [cur_node.x,], [cur_node.y,]
        while cur_node.parent != None:
            x.append(cur_node.parent.x)
            y.append(cur_node.parent.y)
            cur_node = cur_node.parent
        
        # plt.plot(x, y, c = "magenta", label = "Path")

        path[0] += x[::-1]
        path[1] += y[::-1]

        # plt.plot([source_tree.nodes[-1].x, dest_node.x], \
        #          [source_tree.nodes[-1].y, dest_node.y], c = "magenta")

        cur_node = dest_node
        x, y = [cur_node.x,], [cur_node.y,]
        while cur_node.parent != None:
            x.append(cur_node.parent.x)
            y.append(cur_node.parent.y)
            cur_node = cur_node.parent
        
        # plt.plot(x, y, c = "magenta")        

        path[0] += x
        path[1] += y

        plt.plot(path[0], path[1], c = "magenta")
        

        plt.grid(True)
        plt.legend()
        plt.show()

        return path

# y = mx + c
# mx - y + c = 0
# a = m, b = -1, c = c

# y = y1 + m(x - x1)
# c = y1 - mx1

# y = inf x + c
# x + c = 0


# (x - x_c)**2 + (y - y_c)**2 = r**2
# y = mx + c

# (x - x_c)**2 + (mx + c - y_c)**2 = r**2

# x**2 + x_c**2 - 2*x*x_c
# + m**2 * x**2 + (c - y_c)**2 + 2*m*x*(c - y_c)
# - r**2 = 0

# (m**2 + 1)*(x**2) + (2*m*(c - y_c) - 2*x_c)*(x) + (x_c**2 + (c - y_c)**2 - r**2) = 0
    
    def intersects_obj(self, point, node, theta):
        if theta[0] == 0: # not checking for slope = inf. 
            # Simply returning that it will intersect even if it may not.
            # Ease of calculations
            return True
        
        path_bound = 1

        line_eq = (theta[1]/theta[0], -1, point[1] - (theta[1]/theta[0])*point[0])
        for obs in self.obs:
            dist_cen = abs(line_eq[0]*obs.c_x + line_eq[1]*obs.c_y + line_eq[2])/ \
                (line_eq[0]**2 + line_eq[1]**2)**(0.5)

            m = line_eq[0]
            c = line_eq[2]
            x_c = obs.c_x
            y_c = obs.c_y
            r = obs.radius + path_bound

            A = (m**2 + 1)
            B = (2*m*(c - y_c) - 2*x_c)
            C = (x_c**2 + (c - y_c)**2 - r**2)
            
            if dist_cen == r:
                # 3 points -> 1 poi (x) and 2 end points (p)
                # p p x, x p p (no intersection)
                # p x p (intersection)
                x = (-B) / (2*A)
                y = m*x + c

                sorted_colinear_points = sorted([(x, y), point, (node.x, node.y)])

                if sorted_colinear_points[1] == (x, y):
                    return True
                
            elif dist_cen < r:
                # 4 points -> 2 pois (x) and 2 end points (p) of line segment
                # p p x x, x x p p, (no intersection)
                # p x x p, p x p x, x p x p, x p p x (intersection)

                x_1 = (-B - (B**2 - 4*A*C)**(0.5)) / (2*A)
                y_1 = m*x_1 + c
                x_2 = (-B + (B**2 - 4*A*C)**(0.5)) / (2*A)
                y_2 = m*x_2 + c

                sorted_colinear_points = sorted([(x_1, y_1), (x_2, y_2), point, (node.x, node.y)])

                if sorted_colinear_points[0] == (x_1, y_1) and sorted_colinear_points[1] == (x_2, y_2):
                    continue
                elif sorted_colinear_points[2] == (x_1, y_1) and sorted_colinear_points[3] == (x_2, y_2):
                    continue
                else:
                    return True
        return False           

class obstacle:
    def __init__(self, radius, c_x, c_y):
        self.radius = radius
        self.c_x = c_x
        self.c_y = c_y

class node:
    def __init__(self, coord, parent):
        self.x = coord[0]
        self.y = coord[1]
        self.parent = parent
        
class tree:
    def __init__(self, delta, start):
        self.delta = delta
        self.nodes = [node(start, None),]
        self.goal_tree = None
    
    def set_goal(self, goal_tree):
        self.goal_tree = goal_tree
    
    def step(self):
        sample_x = (random.random()-0.5)*100
        sample_y = (random.random()-0.5)*100
        nearest_node, theta = self.find_nearest_node(sample_x, sample_y)
        return (nearest_node.x + self.delta*theta[0], nearest_node.y + self.delta*theta[1]), \
            nearest_node, theta
    
    def find_nearest_node(self, x, y):
        min_dist = 1e10
        min_dist_node = None
        min_dist_dir = None
        for node in self.nodes:
            if dist(node.x, node.y, x, y) < min_dist:
                min_dist = dist(node.x, node.y, x, y)
                min_dist_node = node
                min_dist_dir = ((x - node.x)/min_dist, (y - node.y)/min_dist)
        return min_dist_node, min_dist_dir
    
    def add_node(self, point, parent):
        self.nodes.append(node(point, parent))
    
    def check_merge(self):
        x, y = self.nodes[-1].x, self.nodes[-1].y
        for node in self.goal_tree.nodes:
            if dist(x, y, node.x, node.y) <= self.delta:
                return True, node
        return False, None
        
def main():
    random.seed(42)
    for i in range(3):
        path = map().birrt()
        if path[0][0] == 20 and path[1][0] == 20: # if starting point is 20, 20 -> flip
            path[0] = path[0][::-1]
            path[1] = path[1][::-1]
        savemat(f"path_{i+1}.mat", {"path":np.array(path)})

if __name__ == "__main__":
    main()

    