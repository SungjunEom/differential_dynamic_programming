import numpy as np

class System:
    def __init__(self, cost, horizon, system):
        self.cost = cost
        self.horizon = horizon
        self.inputs = np.random.random(horizon-1)
        self.states = np.zeros(horizon)
        self.system = system

    def optimizer(self, J):
        pass

    def forward(self):
        for i in range(1, len(self.states)):
            self.states[i] = system(self.states[i-1], self.inputs[i-1])
    
    def backward(self):
        J = self.cost(self.states[-1], 0)
        for i in range(len(self.states)-2, 0, -1):
            J = self.optimizer(J)
        

def cost(x, u):
    return x**2 + u**2

def system(x, u):
    return x**3 + (x + 1)*u


if __name__ == "__main__":
    wow = System(cost, 10, system)
    print(wow.states)
    wow.forward()
    print(wow.states)