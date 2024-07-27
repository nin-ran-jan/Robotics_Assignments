import numpy as np
import matplotlib.pyplot as plt

def dist(x_1, y_1, x_2, y_2):
    # capped distance
    return max(((x_1 - x_2)**2 + (y_1 - y_2)**2)**(0.5), 1e-5)

class map():
    def __init__(self, field_type="p", k_att=1, k_rep=1e2, influence_rad=1, gamma=2):
        self.start = (1., 1.)
        self.end = (20., 20.)
        self.obs = [obstacle(2, 4.5, 3), 
                    obstacle(2, 3, 12),
                    obstacle(3, 15, 15)]
        self.field_type = field_type
        self.k_att = k_att
        self.k_rep = k_rep
        self.influence_rad = influence_rad
        self.gamma = gamma
        self.max_iter = 100000
        self.lr = 1e-3
        self.confidence = 1e-3
    
    def get_pot_fields(self, x, y):
        att_pot = 0
        rep_pot = 0
        if self.field_type == "p":
            att_pot = self.k_att*(dist(x, y, self.end[0], self.end[1])**2)
        elif self.field_type == "c":
            att_pot = self.k_att*dist(x, y, self.end[0], self.end[1])
        else:
            print("Wrong field parameter provided!")
            exit(1)
        
        for obs in self.obs:
            if self.in_influence(obs, x, y):
                rep_pot += (self.k_rep/self.gamma)*( ( (1/dist(x, y, obs.c_x, obs.c_y)) - (1/(obs.radius+self.influence_rad)) )**self.gamma)

        return att_pot + rep_pot

    def in_influence(self, obs, x, y):
        if dist(x, y, obs.c_x, obs.c_y) > obs.radius+self.influence_rad:
            return False
        return True

    def get_grad(self, x, y):
        if self.field_type == "p":
            att_pot_grad = (2*self.k_att*(x - self.end[0]), 2*self.k_att*(y - self.end[1]))
        elif self.field_type == "c":
            att_pot_grad = ( (self.k_att*(x - self.end[0]) ) / (dist(x, y, self.end[0], self.end[1])),
                            (self.k_att*(y - self.end[1]) )/ (dist(x, y, self.end[0], self.end[1])) )
        else:
            exit(1)
        
        rep_pot_grad = [0, 0]
        for obs in self.obs:
            if self.in_influence(obs, x, y):
                rep_pot_grad[0] -= self.k_rep*( ( (1/dist(x, y, obs.c_x, obs.c_y)) - \
                                (1/(obs.radius+self.influence_rad)) )**(self.gamma-1) )* \
                                (dist(x, y, obs.c_x, obs.c_y)**3)*(x-obs.c_x)
                rep_pot_grad[1] -= self.k_rep*(((1/dist(x, y, obs.c_x, obs.c_y)) - \
                                (1/(obs.radius+self.influence_rad)))**(self.gamma-1))* \
                                (dist(x, y, obs.c_x, obs.c_y)**3)*(y-obs.c_y)
        
        return (att_pot_grad[0] + rep_pot_grad[0], \
                att_pot_grad[1] + rep_pot_grad[1])

    def solve(self):
        path = [self.start, ]
        for counter in range(self.max_iter):
            grad = self.get_grad(path[-1][0], path[-1][1])
            new_point = (-self.lr*grad[0] + path[-1][0], 
                         -self.lr*grad[1] + path[-1][1])
            path.append(new_point)
            if abs(path[-1][0] - self.end[0]) + abs(path[-1][1] - self.end[1]) < self.confidence:
                print("Converged!")
                self.plot(path)
                break
            
            if counter == self.max_iter-1:
                print("Not converged!")
                self.plot(path)
    
    def plot(self, path):
        x, y = [], []
        for i in path:
            x.append(i[0])
            y.append(i[1])
        plt.plot(x, y, label="Path")
        plt.scatter(self.start[0], self.start[1], label="Start point")
        plt.scatter(self.end[0], self.end[1], label="End point")
        
        angles = np.linspace(0, 2*np.pi, 100)
        plt.plot(self.obs[0].c_x + self.obs[0].radius*np.cos(angles), self.obs[0].c_y + self.obs[0].radius*np.sin(angles), c = "k", label="Obstacles")
        for obs in self.obs[1:]:
            plt.plot(obs.c_x + obs.radius*np.cos(angles), obs.c_y + obs.radius*np.sin(angles), c = "k")
       
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
class obstacle:
    def __init__(self, radius, c_x, c_y):
        self.radius = radius
        self.c_x = c_x
        self.c_y = c_y

def plot_pot_fields(map):
    x, y, pot = [], [], []
    for i in np.arange(-5, 25, 0.1):
        for j in np.arange(-5, 25, 0.1):
            x.append(i)
            y.append(j)
            pot.append(map.get_pot_fields(i, j))
    plt.scatter(x, y, c=pot, cmap='viridis', vmax=1e3, vmin=0)
    plt.colorbar()
    plt.show()

def main():
    for field_type in ["p", "c"]:
        apf_map = map(field_type)
        plot_pot_fields(apf_map)
        apf_map.solve()
    
if __name__ == "__main__":
    main()
    