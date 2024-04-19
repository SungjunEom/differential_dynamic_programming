import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class System:
    def __init__(self, loss, horizon, sys, x0, xN, dloss, dsys):
        self.loss = loss
        self.horizon = horizon
        self.dloss = dloss
        self.dsys = dsys
        self.inputs = np.zeros(horizon-1)
        self.states = np.zeros(horizon)
        self.states[0] = x0
        self.state_dest = xN
        self.delta_states = np.zeros(horizon)
        self.sys = sys
        self.sum_count = 0

    def get_states(self):
        return self.states
    
    def set_dest(self, new=None):
        if new is None:
            # self.states[-1] = self.state_dest
            self.states[-1] = -self.state_dest
        else:
            # self.state_dest = new
            self.state_dest = -new

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
            self.delta_states[i] = state - self.states[i]
            self.states[i] = state
    
    def backward_numerical(self):
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
            
            self.Quu = Quu

            # k = -Qu/(Quu+1e-10)
            # K = -Qux/(Quu+1e-10)
            k = -Qu/Quu
            K = -Qux/Quu
            Vx = Qx - K*Quu*k
            Vxx = Qxx - K*Quu*K

            self.inputs[i] = self.inputs[i] + (k + K * self.delta_states[i])

    def backward(self):
        self.set_dest()
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
    
    def full_cost(self, loss):
        sum = 0
        for i in range(self.horizon-1):
            sum += loss(self.states[i], self.inputs[i])
        sum += loss(self.states[-1], 0)
        return sum
    
    def get_Quu(self):
        return self.Quu
        

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
    sys2 = System(loss,100, system, 0.1, 2, dloss, dsystem)
    errors2 = []
    states2 = []
    cost2 = []
    Quu2 = []
    for i in range(150):
        sys2.backward()
        sys2.forward()
        states2.append(sys2.states[-1])
        Quu2.append(sys2.get_Quu())
        errors2.append(sys2.error())
        cost2.append(sys2.full_cost(loss))
        if i > 100:
            sys2.set_dest(0.3)
        elif i > 50:
            sys2.set_dest(0.1)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(errors2, label='x0=0.2, xN=0.5')
    axs[0, 0].set_title('state errors')
    axs[0, 1].plot(cost2, label='x0=0.2, xN=0.5')
    axs[0, 1].set_title('Full cost')
    axs[1, 0].plot(Quu2, label='x0=0.2, xN=0.5')
    axs[1, 0].set_title('Quu')
    axs[1, 1].plot(states2, label='x0=0.2, xN=0.5')
    axs[1, 1].set_title('States')
    plt.show()