import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class System:
    def __init__(self, loss, horizon, sys, x0, dloss, dsys):
        self.loss = loss
        self.horizon = horizon
        self.dloss = dloss
        self.dsys = dsys
        self.inputs = np.zeros(horizon-1)
        self.states = np.zeros(horizon)
        self.states[0] = x0
        self.delta_states = np.zeros(horizon)
        self.sys = sys
        self.ks = np.zeros(horizon-1)
        self.Ks = np.zeros(horizon-1)
        self.sum_count = 0

    def forward(self):
        for i in range(1, len(self.states)):
            state = self.sys(self.states[i-1], self.inputs[i-1])
            self.delta_states[i] = state - self.states[i]
            self.inputs[i-1] = self.inputs[i-1] + (self.ks[i-1]+self.Ks[i-1]*self.delta_states[i])
            self.states[i] = state
    
    def backward(self):
        Vx = self.dloss('x', self.states[-1], 0)
        Vxx = self.dloss('xx', self.states[-1], 0)
        for i in range(len(self.states)-2, -1, -1):
            Qx = self.dloss('x', self.states[i], self.inputs[i]) \
                + self.dsys('x', self.states[i], self.inputs[i]) \
                * Vx
            Qu = self.dloss('u', self.states[i], self.inputs[i]) \
                + self.dsys('u', self.states[i], self.inputs[i]) \
                * Vx
            Qxx = self.dloss('xx', self.states[i], self.inputs[i]) \
                + self.dsys('x', self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.dsys('xx', self.states[i], self.inputs[i])
            Qux = self.dloss('ux', self.states[i], self.inputs[i]) \
                + self.dsys('u', self.states[i], self.inputs[i]) \
                * Vxx * self.dsys('x', self.states[i], self.inputs[i]) \
                + Vx * self.dsys('ux', self.states[i], self.inputs[i])
            Quu = self.dloss('uu', self.states[i], self.inputs[i]) \
                + self.dsys('u', self.states[i], self.inputs[i])**2 \
                * Vxx \
                + Vx * self.dsys('uu', self.states[i], self.inputs[i])
            
            self.Quu = Quu

            # k = -Qu/(Quu+1e-10)
            # K = -Qux/(Quu+1e-10)
            k = -Qu/Quu
            K = -Qux/Quu

            self.ks[i]= k
            self.Ks[i] = K

            # Vx = Qx - K*Quu*k
            # Vxx = Qxx - K*Quu*K
            Vx = Qx - Qux*Qu/Quu
            Vxx = Qxx - Qux*Qux/Quu
    
    def full_cost(self, loss):
        sum = 0
        for i in range(self.horizon-1):
            sum += loss(self.states[i], self.inputs[i])
        sum += loss(self.states[-1], 0)
        return sum
            

def loss(x, u):
    return x**2 + u**2

def dloss(target, x, u):
    if target=='x':
        return 2*x
    elif target=='xx':
        return 2
    elif target=='u':
        return 2*u
    elif target=='uu':
        return 2
    elif target=='ux':
        return 0

def system(x,u):
    return x**3 + (x**2 + 1)*u

def dsystem(target, x, u):
    if target=='x':
        return 3*x**2+2*x*u
    elif target=='xx':
        return 6*x+2*u
    elif target=='u':
        return x**2+1
    elif target=='uu':
        return 0
    elif target=='ux':
        return 2*x


if __name__ == "__main__":
    sys2 = System(loss,5, system, 1.5, dloss, dsystem)
    for i in range(10):
        sys2.backward()
        sys2.forward()
        print('ks: ',sys2.ks)
        print('Ks: ', sys2.Ks)

    print('Quu:',sys2.Quu)
    states = sys2.states
    plt.plot(states)
    inputs = sys2.inputs
    plt.plot(inputs)
    plt.show()