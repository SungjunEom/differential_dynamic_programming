import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class System:
    def __init__(self, loss, horizon, sys, x0, xN):
        self.loss = loss
        self.horizon = horizon
        self.inputs = np.zeros(horizon-1)
        self.states = np.zeros(horizon)
        self.states[0] = x0
        self.state_dest = xN
        self.delta_states = np.zeros(horizon)
        self.sys = sys
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
            state = self.sys(self.states[i-1], self.inputs[i-1])
            self.delta_states[i] = self.states[i] - state
            self.states[i] = state
    
    def backward(self):
        self.set_dest()
        Vx = self.diff_x(self.loss, self.states[-1], 0)
        Vxx = self.diff_xx(self.loss, self.states[-1],0)
        for i in range(len(self.states)-2, -1, -1):
            Qx = self.diff_x(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_x(self.sys, self.states[i], self.inputs[i]) \
                * Vx
            Qu = self.diff_u(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.sys, self.states[i], self.inputs[i]) \
                * Vx
            Qxx = self.diff_xx(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_x(self.sys, self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.diff_xx(self.sys, self.states[i], self.inputs[i])
            Qux = self.diff_ux(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.sys, self.states[i], self.inputs[i]) \
                * Vxx * self.diff_x(self.sys, self.states[i], self.inputs[i]) \
                + Vx * self.diff_ux(self.sys, self.states[i], self.inputs[i])
            Quu = self.diff_uu(self.loss, self.states[i], self.inputs[i]) \
                + self.diff_u(self.sys, self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.diff_uu(self.sys, self.states[i], self.inputs[i])
            
            k = -Qu/(Quu+1e-10)
            K = -Qux/(Quu+1e-10)
            # k = -Qu/Quu
            # K = -Qux/Quu
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
    sys2 = System(loss, 10, system, 0.9, 0.5)
    sys3 = System(loss, 10, system, 0.1, 0.2)
    errors2 = []
    errors3 = []
    for i in range(50):
        sys2.backward()
        sys3.backward()
        sys2.forward()
        sys3.forward()
        errors2.append(sys2.error())
        errors3.append(sys3.error())
    plt.plot(errors2, label='x0=0.9, xN=0.5')
    plt.plot(errors3, label='x0=0.1, xN=0.2')
    plt.ylabel('error of the final states')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()