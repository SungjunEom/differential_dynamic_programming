import numpy as np

class System:
    def __init__(self, loss, horizon, system):
        self.loss = loss
        self.horizon = horizon
        self.inputs = np.random.random(horizon-1)
        self.states = np.zeros(horizon)
        self.delta_states = np.zeros(horizon)
        self.system = system

    def diff_x(self, func, x0, u0, delta=1e-7):
        return (func(x0+delta, u0)-func(x0, u0))/delta

    def diff_u(self, func, x0, u0, delta=1e-7):
        return (func(x0, u0+delta)-func(x0, u0))/delta
    
    def diff_xx(self, func, x0, u0, delta=1e-7):
        return (self.diff_x(func, x0+delta, u0) - self.diff_x(func, x0, u0))/delta
    
    def diff_uu(self, func, x0, u0, delta=1e-7):
        return (self.diff_u(func, x0, u0+delta) - self.diff_u(func, x0, u0))/delta
    
    def diff_xu(self, func, x0, u0, delta=1e-7):
        return (self.diff_x(func, x0, u0+delta) - self.diff_x(func, x0, u0))/delta
        
    def diff_ux(self, func, x0, u0, delta=1e-7):
        return (self.diff_u(func, x0+delta, u0) - self.diff_u(func, x0, u0))/delta
    
    def optimizer(self, iter):
        Quu_k = self.diff_uu(self.loss(self.states[iter], self.inputs[iter])) \
                + self.diff_u(self.system(self.inputs[iter]))

    def forward(self):
        for i in range(1, len(self.states)):
            state =  system(self.states[i-1], self.inputs[i-1])
            self.delta_states[i] = self.states[i] - state
            self.states[i] = state
    
    def backward(self):
        J = self.loss(self.states[-1], 0)
        for i in range(len(self.states)-2, 0, -1):
            pass
        

def loss(x, u):
    return x**2 + u**2

def system(x, u):
    return x**3 + (x + 1)*u


if __name__ == "__main__":
    wow = System(loss, 10, system)
    print(wow.diff_u(wow.loss, 9, 10))