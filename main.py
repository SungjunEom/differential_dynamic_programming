import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class System:
    def __init__(self, loss, horizon, system, x0, xN):
        self.loss = loss
        self.horizon = horizon
        # self.inputs = np.random.random(horizon-1)
        self.inputs = np.zeros(horizon-1)
        self.states = np.zeros(horizon)
        self.states[0] = x0
        self.state_dest = xN
        self.delta_states = np.zeros(horizon)
        self.system = system
        self.sum_count = 0

    def get_states(self):
        return self.states
    
    def set_dest(self):
        self.states[-1] = self.state_dest

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

    def forward(self):
        for i in range(1, len(self.states)):
            state = self.system(self.states[i-1], self.inputs[i-1])
            self.delta_states[i] = self.states[i] - state
            self.states[i] = state
    
    def backward(self):
        self.set_dest()
        Vx = self.diff_x(self.loss, self.states[-1], 0)
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
                * Vxx \
                + Vx * self.diff_uu(self.system, self.states[i], self.inputs[i])
            
            k = -Qu/Quu
            K = -Qux/Quu
            Vx = Qx - K*Quu*k
            Vxx = Qxx - K*Quu*K

            self.inputs[i] = self.inputs[i] + (k + K * self.delta_states[i])

    # Auxiliaries
    def summary(self):
        self.sum_count += 1
        print()
        print('====== Summary',self.sum_count,'======')
        print('states:',self.states)
        print('inputs:',self.inputs)
        print('=======================')

    def error(self):
        return self.states[-1] - self.state_dest
            
        

def loss(x, u):
    return x**2 + u**2

def system(x, u):
    return x**3 + (x**2 + 1)*u


if __name__ == "__main__":
    wow = System(loss, 10, system, 0.2, 0.3)
    errors = []
    for i in range(100):
        wow.forward()
        errors.append(wow.error())
        wow.backward()
    print(errors)
    plt.plot(errors)
    plt.ylabel('error of the final states')
    plt.xlabel('iteration')
    plt.show()