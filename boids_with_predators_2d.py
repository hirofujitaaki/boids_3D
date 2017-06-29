import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm

WIDTH, HEIGHT = 640, 480

MIN_DIST = 25.0
MAX_RULE_VEL = 0.03
MAX_VEL = 2.0

b_num = 100
p_num = 1


class Boids:
    """Class that represents Boids simulation"""
    def __init__(self, b_num):
        """ initialize the Boid simulation"""
        self.pos = [WIDTH/2.0, HEIGHT/2.0] + 10*np.random.rand(2*b_num).reshape(b_num, 2)
        # array([[ 324.45589932,  241.17939184],
        #        [ 322.87078634,  248.77799187],
        #        [ 324.33537195,  242.2450071 ],
        #        [ 327.11461049,  245.12112812]])  #e.g.

        angles = 2*math.pi*np.random.rand(b_num)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))
        # array([[ 0.19352404, -0.98109553],
        #        [ 0.85258769, -0.52258419],
        #        [ 0.55343486,  0.83289246],
        #        [-0.95993835,  0.28021128]])  #e.g.

        # number of boids
        self.b_num = b_num
        # min dist of approach
        self.min_dist = MIN_DIST
        # max magnitude of velocities calculated by "rules"
        self.max_rule_vel = MAX_RULE_VEL
        # max maginitude of final velocity
        self.max_vel = MAX_VEL

    def tick(self, frame_num, pts, beak):
        """Update the simulation by one time step."""
        # get pairwise distances
        self.dist_matrix = squareform(pdist(self.pos))
        # array([[ 0.        ,  7.76217144,  1.07240977,  4.75457989],
        #        [ 7.76217144,  0.        ,  6.6951401 ,  5.60202605],
        #        [ 1.07240977,  6.6951401 ,  0.        ,  3.99952985],
        #        [ 4.75457989,  5.60202605,  3.99952985,  0.        ]])  #e.g

        # apply rules:
        self.vel += self.apply_rules()
        self.limit(self.vel, self.max_vel)
        self.pos += self.vel
        self.apply_bc()
        # update data
        pts.set_data(self.pos.reshape(2*self.b_num)[::2],  # picks out odd-numbered elements
                     self.pos.reshape(2*self.b_num)[1::2]) # picks out even-numbered elements
        vec = self.pos + 10*self.vel/self.max_vel
        beak.set_data(vec.reshape(2*self.b_num)[::2],
                      vec.reshape(2*self.b_num)[1::2])

    def limit_vec(self, vec, max_val):
        """limit magnitide of 2D vector"""
        mag = norm(vec)
        if mag > max_val:
            vec[0], vec[1] = vec[0]*max_val/mag, vec[1]*max_val/mag

    def limit(self, X, max_val):
        """limit magnitide of 2D vectors in array X to max_value"""
        for vec in X:
            self.limit_vec(vec, max_val)

    def apply_bc(self):
        """apply boundary conditions"""
        DELTA_R = 2.0
        for coord in self.pos:
            if coord[0] > WIDTH + DELTA_R:
                coord[0] = - DELTA_R
            if coord[0] < - DELTA_R:
                coord[0] = WIDTH + DELTA_R
            if coord[1] > HEIGHT + DELTA_R:
                coord[1] = - DELTA_R
            if coord[1] < - DELTA_R:
                coord[1] = HEIGHT + DELTA_R

    def apply_rules(self):
        # apply rule #1 - Separation
        D = self.dist_matrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.b_num, 1) - D.dot(self.pos)
        self.limit(vel, self.max_rule_vel)

        # different distance threshold
        D = self.dist_matrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.max_rule_vel)
        vel += vel2

        # apply rule #1 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.max_rule_vel)
        vel += vel3

        return vel


class Predators():

    def __init__(self, p_num=1):
        """ initialize the Boid simulation"""

        self.pos = [WIDTH/2.0, HEIGHT/2.0] + np.random.uniform(-200, 200, 2)
            # array([[ 324.45589932,  241.17939184],
            #        [ 322.87078634,  248.77799187],
            #        [ 324.33537195,  242.2450071 ],
            #        [ 327.11461049,  245.12112812]])  #e.g.

        angles = 2*math.pi*np.random.rand(p_num)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))
        # array([[ 0.19352404, -0.98109553],
        #        [ 0.85258769, -0.52258419],
        #        [ 0.55343486,  0.83289246],
        #        [-0.95993835,  0.28021128]])  #e.g.


        # number of boids
        self.p_num = p_num
        # min dist of approach
        self.min_dist = MIN_DIST
        # max magnitude of velocities calculated by "rules"
        self.max_rule_vel = MAX_RULE_VEL
        # max maginitude of final velocity
        self.max_vel = MAX_VEL

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
        p_body.set_data(self.pos.reshape(2*self.p_num)[::2],  # picks out odd-numbered elements
                     self.pos.reshape(2*self.p_num)[1::2]) # picks out even-numbered elements
        # vec = self.pos + 10*self.vel/self.maxvel
        # beak.set_data(vec.reshape(2*self.b_num)[::2],
        #               vec.reshape(2*self.b_num)[1::2])

    def limit_vec(self, vec, max_val):
        """limit magnitide of 2D vector"""
        mag = norm(vec)
        if mag > max_val:
            vec[0], vec[1] = vec[0]*max_val/mag, vec[1]*max_val/mag

    def limit(self, X, max_val):
        """limit magnitide of 2D vectors in array X to max_value"""
        for vec in X:
            self.limit_vec(vec, max_val)

    def apply_bc(self):
        """apply boundary conditions"""
        DELTA_R = 2.0
        for coord in self.pos:
            if coord[0] > WIDTH + DELTA_R:
                coord[0] = - DELTA_R
            if coord[0] < - DELTA_R:
                coord[0] = WIDTH + DELTA_R
            if coord[1] > HEIGHT + DELTA_R:
                coord[1] = - DELTA_R
            if coord[1] < - DELTA_R:
                coord[1] = HEIGHT + DELTA_R

    def apply_rules(self):
        # apply rule #1 - Separation
        D = self.dist_matrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.b_num, 1) - D.dot(self.pos)
        self.limit(vel, self.max_rule_vel)

        # different distance threshold
        D = self.dist_matrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.max_rule_vel)
        vel += vel2

        # apply rule #1 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.max_rule_vel)
        vel += vel3

        return vel

    def button_press(self, event):
        """event handler for matplotlib button presses"""
        # left click - add a boid
        if event.button is 1:
            self.pos = np.concatenate((self.pos,
                          np.array([[event.xdata, event.ydata]])), axis=0)
            # random velocity
            angles = 2*math.pi*np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.b_num += 1


def tick(frame_num, pts, beak, boids, p_body, predators):
    """update function for animation"""
    boids.tick(frame_num, pts, beak)
    predators.tick(frame_num, p_body)
    return pts, beak, p_body


def main():
    print('starting boids...')

    boids = Boids(b_num)
    predators = Predators(p_num)

    # setup plot
    fig = plt.figure()
    ax = plt.axes(xlim=(0, WIDTH), ylim=(0, HEIGHT))

    # add a "button press" event handler
    cid = fig.canvas.mpl_connect('button_press_event', predators.button_press)

    # ax.plot() returns a tuple with one element like:
    # ([x1, x2, x3, x4], [y1, y2, y3, y4]),
    pts, = ax.plot([], [], markersize=10, c='w', marker='o', ls='None')
    beak, = ax.plot([], [], markersize=4, c='y', marker='o', ls='None')
    p_body, = ax.plot([], [], markersize=15, c='#a3631f', marker='o', ls='None')
    anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids, p_body,
                                                     predators), interval=50)

    plt.show()


if __name__ == '__main__':
    main()
