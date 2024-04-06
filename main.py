import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

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
    
    def optimizer(self, k, K):
        for i in range(len(self.inputs)):
            self.inputs[i] = self.inputs[i] + (k + K * self.delta_states[i])

    def forward(self):
        for i in range(1, len(self.states)):
            state = system(self.states[i-1], self.inputs[i-1])
            self.delta_states[i] = self.states[i] - state
            self.states[i] = state
    
    def backward(self):
        Vx = self.diff_x(self.loss,self.states[-1],0)
        Vxx = self.diff_xx(self.loss, self.states[-1],0)
        for i in range(len(self.states)-2, -1, -1):
            Qx = self.diff_x(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_x(self.system, self.states[i], self.inputs[i]) \
                * Vx
            Qu = self.diff_u(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.system, self.states[i], self.inputs[i]) \
                * Vx
            Qxx = self.diff_xx(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_x(self.system, self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.diff_xx(self.system, self.states[i], self.inputs[i])
            Qux = self.diff_ux(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.system, self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.diff_ux(self.system, self.states[i], self.inputs[i])
            Quu = self.diff_uu(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.system, self.states[i], self.inputs[i])**2 \
                + Vx * self.diff_uu(self.system, self.states[i], self.inputs[i])
            
            k = -Qu/Quu
            K = -Qux/Quu
            Vx = Qx - K*Quu*k
            Vxx = Qxx - K*Quu*K

    def summary(self):
        print(self.states)
            
        

def loss(x, u):
    return x**2 + u**2

def system(x, u):
    return x**3 + (x + 1)*u


if __name__ == "__main__":
    wow = System(loss, 10, system)
    wow.summary()
    wow.forward()
    wow.summary()
    wow.backward()
    wow.summary()
    wow.forward()
    wow.summary()
    wow.backward()
    wow.summary()
    plt.plot(wow.states)
    plt.yscale('log', base=10)
    plt.show()