import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class System:
    def __init__(self, loss, horizon, sys, x0, dloss, dsys, dumax=0.6):
        self.loss = loss
        self.horizon = horizon
        self.dloss = dloss
        self.dsys = dsys
        self.x0 = x0
        self.inputs = np.zeros(horizon-1)
        self.states = np.zeros(horizon)
        self.states[0] = x0
        self.delta_states = np.zeros(horizon)
        self.sys = sys
        self.ks = np.zeros(horizon-1)
        self.Ks = np.zeros(horizon-1)
        self.sum_count = 0
        self.dumax = dumax
        self.v_bar = np.inf
        self.gamma_flag = False # For a debugging purpose only. Can be deleted.

    def forward(self):
        x_hat = np.zeros(self.horizon-1)
        u_hat = np.zeros(self.horizon-1)
        x_hat[0] = self.x0
        u_hat[0] = self.ks[0]
        for j in range(50): # Maximum gamma iteration
            for i in range(len(self.inputs)-1):
                x_hat[i+1] = self.sys(x_hat[i], u_hat[i])
                dcontrol_i = self.ks[i+1]*pow(1/2,j) + self.Ks[i+1]*(x_hat[i+1] - self.states[i+1])
                u_hat[i+1] = self.inputs[i+1] + dcontrol_i
            v = self.full_cost(x_hat, u_hat)
            if v < self.v_bar:
                print('v_bar: ', self.v_bar)
                print('v: ', v)
                self.v_bar = v
                break
            print('Gamma is being updated')
            self.gamma_flag = True
            
        self.states = x_hat
        self.inputs = u_hat

    
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

            # This system has a scalar input. Always be the restriction.
            Quu = max(Quu, abs(Quu)/self.dumax)
            k = -Qu/Quu
            K = -Qux/Quu

            self.ks[i]= k
            self.Ks[i] = K

            # Vx = Qx - K*Quu*k
            # Vxx = Qxx - K*Quu*K
            Vx = Qx - Qux*Qu/Quu
            Vxx = Qxx - Qux*Qux/Quu
    
    def full_cost(self, x=None, u=None):
        sum = 0
        if x is not None and u is not None:
            for i in range(self.horizon-1):
                sum += self.loss(x[i], u[i])
            sum += self.loss(x[-1], 0)
        else:
            for i in range(self.horizon-1):
                sum += self.loss(self.states[i], self.inputs[i])
            sum += self.loss(self.states[-1], 0)
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
    sys2 = System(loss, 6, system, 1.02, dloss, dsystem)
    states = []
    for i in range(100):
        print('iteration: ', i)
        sys2.backward()
        sys2.forward()
        # print('ks: ',sys2.ks)
        # print('Ks: ', sys2.Ks)
        states.append(sys2.states)

    print('Quu:',sys2.Quu)
    # plt.plot(states[9], label='10th iteration')
    plt.plot(states[19], label='20th iteration')
    plt.plot(states[29], label='30th iteration')
    plt.plot(states[49], label='50th iteration')
    plt.plot(states[59], label='60th iteration')
    plt.plot(states[-1], label='100th iteration')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.show()

    plt.plot(sys2.inputs, 'ko-', label='final_inputs')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Input')
    plt.show()
    print(sys2.inputs)
    print(sys2.gamma_flag)