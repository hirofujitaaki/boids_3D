
"""
boids.py
Implementation of Craig Reynold's BOIDs
Author: Mahesh Venkitachalam
"""

import sys, argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

width, height = 640, 480


class Boids:
    """Class that represents Boids simulation"""
    def __init__(self, N):
        """ initialize the Boid simulation"""
        self.pos = [width/2.0, height/2.0] + 10*np.random.rand(2*N).reshape(N, 2)
        # array([[ 324.45589932,  241.17939184],
        #        [ 322.87078634,  248.77799187],
        #        [ 324.33537195,  242.2450071 ],
        #        [ 327.11461049,  245.12112812]])  #e.g.

        angles = 2*math.pi*np.random.rand(N)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))
        # array([[ 0.19352404, -0.98109553],
        #        [ 0.85258769, -0.52258419],
        #        [ 0.55343486,  0.83289246],
        #        [-0.95993835,  0.28021128]])  #e.g.

        # number of boids
        self.N = N
        # min dist of approach
        self.minDist = 25.0
        # max magnitude of velocities calculated by "rules"
        self.maxRuleVel = 0.03
        # max maginitude of final velocity
        self.maxVel = 2.0

    def tick(self, frame_num, pts, beak):
        """Update the simulation by one time step."""
        # get pairwise distances
        self.distMatrix = squareform(pdist(self.pos))
        # array([[ 0.        ,  7.76217144,  1.07240977,  4.75457989],
        #        [ 7.76217144,  0.        ,  6.6951401 ,  5.60202605],
        #        [ 1.07240977,  6.6951401 ,  0.        ,  3.99952985],
        #        [ 4.75457989,  5.60202605,  3.99952985,  0.        ]])  #e.g

        # apply rules:
        self.vel += self.apply_rules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.apply_bc()
        # update data
        pts.set_data(self.pos.reshape(2*self.N)[::2],  # picks out odd-numbered elements
                     self.pos.reshape(2*self.N)[1::2]) # picks out even-numbered elements
        vec = self.pos + 10*self.vel/self.maxVel
        beak.set_data(vec.reshape(2*self.N)[::2],
                      vec.reshape(2*self.N)[1::2])

    def limit_vec(self, vec, maxVal):
        """limit magnitide of 2D vector"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag

    def limit(self, X, maxVal):
        """limit magnitide of 2D vectors in array X to maxValue"""
        for vec in X:
            self.limit_vec(vec, maxVal)

    def apply_bc(self):
        """apply boundary conditions"""
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR

    def apply_rules(self):
        # apply rule #1 - Separation
        D = self.distMatrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        # different distance threshold
        D = self.distMatrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2;

        # apply rule #1 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel


class Predators():

    def __init__(self, P_NUM=1):
        """ initialize the Boid simulation"""

        self.pos = [width/2.0, height/2.0] + np.random.uniform(100, 100, 1)
            # array([[ 324.45589932,  241.17939184],
            #        [ 322.87078634,  248.77799187],
            #        [ 324.33537195,  242.2450071 ],
            #        [ 327.11461049,  245.12112812]])  #e.g.

        angles = 2*math.pi*np.random.rand(P_NUM)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))
        # array([[ 0.19352404, -0.98109553],
        #        [ 0.85258769, -0.52258419],
        #        [ 0.55343486,  0.83289246],
        #        [-0.95993835,  0.28021128]])  #e.g.


        # number of boids
        self.P_NUM = P_NUM
        # min dist of approach
        self.minDist = 25.0
        # max magnitude of velocities calculated by "rules"
        self.maxRuleVel = 0.03
        # max maginitude of final velocity
        self.maxVel = 2.0

    def tick(self, frame_num, p_body):
        """update the simulation by one time step."""
        # get pairwise distances
        # self.distmatrix = squareform(pdist(self.pos))
        # array([[ 0.        ,  7.76217144,  1.07240977,  4.75457989],
        #        [ 7.76217144,  0.        ,  6.6951401 ,  5.60202605],
        #        [ 1.07240977,  6.6951401 ,  0.        ,  3.99952985],
        #        [ 4.75457989,  5.60202605,  3.99952985,  0.        ]])  #e.g

        # apply rules:
        # self.vel += self.apply_rules()
        # self.limit(self.vel, self.maxvel)
        # self.pos += self.vel
        # self.apply_bc()
        # update data
        p_body.set_data(self.pos.reshape(2*self.P_NUM)[::2],  # picks out odd-numbered elements
                     self.pos.reshape(2*self.P_NUM)[1::2]) # picks out even-numbered elements
        # vec = self.pos + 10*self.vel/self.maxvel
        # beak.set_data(vec.reshape(2*self.N)[::2],
        #               vec.reshape(2*self.N)[1::2])

    def limit_vec(self, vec, maxVal):
        """limit magnitide of 2D vector"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag

    def limit(self, X, maxVal):
        """limit magnitide of 2D vectors in array X to maxValue"""
        for vec in X:
            self.limit_vec(vec, maxVal)

    def apply_bc(self):
        """apply boundary conditions"""
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR

    def apply_rules(self):
        # apply rule #1 - Separation
        D = self.distMatrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        # different distance threshold
        D = self.distMatrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2;

        # apply rule #1 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel

    def button_press(self, event):
        """event handler for matplotlib button presses"""
        # left click - add a boid
        if event.button is 1:

            self.pos = np.concatenate((self.pos,
                                       np.array([[event.xdata, event.ydata]])),
                                      axis=0)
            # random velocity
            angles = 2*math.pi*np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.N += 1
        # right click - scatter
        elif event.button is 3:
            # add scattering velocity
            self.vel += 0.1*(self.pos - np.array([[event.xdata, event.ydata]]))

def tick(frame_num, pts, beak, boids, p_body, predators):
    """update function for animation"""
    boids.tick(frame_num, pts, beak)
    predators.tick(frame_num, p_body)
    return pts, beak, p_body


def main():
    print('starting boids...')

    parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")
    parser.add_argument('--num-boids', dest='N', required=False)
    args = parser.parse_args()

    N = 100
    if args.N:
        N = int(args.N)

    boids = Boids(N)

    P_NUM = 1
    predators = Predators(P_NUM)

    # setup plot
    fig = plt.figure()
    ax = plt.axes(xlim=(0, width), ylim=(0, height))

    # add a "button press" event handler
    cid = fig.canvas.mpl_connect('button_press_event', predators.button_press)

    # ax.plot() returns a tuple with one element like:
    # ([x1, x2, x3, x4], [y1, y2, y3, y4]),
    pts, = ax.plot([], [], markersize=10,
                      c='b', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4,
                      c='k', marker='o', ls='None')
    p_body, = ax.plot([], [], markersize=15,
                      c='r', marker='o', ls='None')
    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids, p_body,
                                                     predators), interval=50)


    plt.show()

if __name__ == '__main__':
  main()
