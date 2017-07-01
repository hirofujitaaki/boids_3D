import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

WIDTH, HEIGHT         = 640, 480
MIN_DIST              = 25.0
MAX_RULE_VEL          = 0.03
MAX_VEL               = 2.0
BOID_RADIUS          = 75.0
PREDATOR_RADIUS       = 100.0
WEIGHT_AVOID_PREDATOR = 100


class Birds():
    def __init__(self, num):
        self.num = num
        # min dist of approach
        self.min_dist = MIN_DIST
        # max magnitude of velocities calculated by "rules"
        self.max_rule_vel = MAX_RULE_VEL
        # max maginitude of final velocity
        self.max_vel = MAX_VEL

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


class Boids(Birds):
    """Class that represents Boids simulation"""

    def __init__(self, num):
        """ initialize the Boid simulation"""
        super().__init__(num)
        self.pos = [WIDTH/2.0, HEIGHT/2.0] + 10*np.random.rand(2*num).reshape(num, 2)
        angles = 2*math.pi*np.random.rand(num)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))

    def tick(self, frame_num, pts, beak, predators):
        """Update the simulation by one time step."""
        # get pairwise distances
        self.b2b_dist_matrix = squareform(pdist(self.pos))

        self.b2p_dist_matrix = cdist(self.pos, predators.pos)
        # returns like: array([[ 165.66129049,   86.37653115],
        #                      [ 123.7833912 ,   44.10185375],
        #                      [ 191.70244957,   60.83571559]]) if 3 boids and 2 predators

        # apply rules:
        self.vel += self.apply_rules(predators)
        self.limit(self.vel, self.max_vel)
        self.pos += self.vel
        self.apply_bc()

        # update data
        pts.set_data(self.pos.reshape(2*self.num)[::2],
                     self.pos.reshape(2*self.num)[1::2])
        vec = self.pos + 10*self.vel/self.max_vel
        beak.set_data(vec.reshape(2*self.num)[::2],
                      vec.reshape(2*self.num)[1::2])

    def apply_rules(self, predators):
        # apply rule #1 - Separation
        D = self.b2b_dist_matrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.num, 1) - D.dot(self.pos)
        self.limit(vel, self.max_rule_vel)

        # different distance threshold
        D = self.b2b_dist_matrix < 50.0

        # apply rule #2 - Alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.max_rule_vel)
        vel += vel2

        # apply rule #3 - Cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.max_rule_vel)
        vel += vel3

        # extra rule: avoid the predators
        D = self.b2p_dist_matrix < BOID_RADIUS

        # calculates the boid-to-predators displacements and returns the value
        # if within the RADIUS
        dist = np.empty((0,2), int)  # has to be (0, n) NOT (1, n)
        for i in range(self.num):
            for j in range(predators.num):
                if D[i][j]:
                    vel_tmp = (self.pos[i] - predators.pos[j]).reshape(1,2)
                else:
                    vel_tmp = np.zeros((1,2))
                dist = np.append(dist, vel_tmp, axis=0)

        # reshapes the array and adds up Xs and Ys per each boid
        vel4 = np.empty((0,2), int)
        dist = dist.reshape(self.num, predators.num*2)
        for i in range(self.num):
            vel_x = np.sum(dist[i][::2])
            vel_y = np.sum(dist[i][1::2])
            vel4 = np.append(vel4, np.array([[vel_x, vel_y]]), axis=0)

        self.limit(vel4, self.max_rule_vel)
        vel4 = vel4*WEIGHT_AVOID_PREDATOR
        vel += vel4

        return vel


class Predators(Birds):
    """Class that represents Predators simulation"""

    def __init__(self, num):
        """ initialize the Boid simulation"""
        super().__init__(num)

        self.pos = [WIDTH/2.0, HEIGHT/2.0] + np.random.uniform(-180, 180, num*2).reshape(num, 2)
        # should return like: array([[ 324.45589932,  241.17939184]])
        # without .reshape(): array([ 322.87078634,  248.77799187])

        angles = 2*math.pi*np.random.rand(num)
        self.vel = np.array(list(zip(np.cos(angles), np.sin(angles))))

    def tick(self, frame_num, p_body, boids):
        """update the simulation by one time step."""
        # get pairwise distances
        self.p2b_dist_matrix = cdist(self.pos, boids.pos)

        # apply rules:
        self.vel += self.apply_rules(boids)
        self.limit(self.vel, self.max_vel)
        self.pos += self.vel
        self.apply_bc()

        # update data
        p_body.set_data(self.pos.reshape(2*self.num)[::2],
                     self.pos.reshape(2*self.num)[1::2])

    def apply_rules(self, boids):

        # apply rule 1: Steer to move towards the center of mass
        mass_center = np.sum(boids.pos, axis=0) / boids.num
        vel = mass_center - self.pos
        self.limit(vel, self.max_rule_vel)


        # apply rule 2: Chase the boids with in the range
        D = self.p2b_dist_matrix < PREDATOR_RADIUS

        # calculates the predator-to-boids displacements and returns the value
        # if within the RADIUS
        dist = np.empty((0,2), int)  # has to be (0, n) NOT (1, n)
        for i in range(self.num):
            for j in range(boids.num):
                if D[i][j]:
                    vel_tmp = (boids.pos[j] - self.pos[i]).reshape(1,2)
                else:
                    vel_tmp = np.zeros((1,2))
                dist = np.append(dist, vel_tmp, axis=0)

        # reshapes the array and adds up Xs and Ys per each boid
        vel2 = np.empty((0,2), int)
        dist = dist.reshape(self.num, boids.num*2)
        for i in range(self.num):
            vel_x = np.sum(dist[i][::2])
            vel_y = np.sum(dist[i][1::2])
            vel2 = np.append(vel2, np.array([[vel_x, vel_y]]), axis=0)

        # import pdb; pdb.set_trace()
        self.limit(vel2, self.max_rule_vel)
        vel += vel2

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
            self.num += 1


def tick(frame_num, pts, beak, boids, p_body, predators):
    """update function for animation"""
    boids.tick(frame_num, pts, beak, predators)
    predators.tick(frame_num, p_body, boids)
    return pts, beak, p_body


def main():
    print('starting boids...')

    # create boids and predators objects
    boids_num = 11
    predators_num = 1
    boids = Boids(boids_num)
    predators = Predators(predators_num)

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
