"""
Path tracking simulation with Stanley steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 65  # [m] Wheel base of vehicle
max_steer = np.radians(20.0)  # [rad] max steering angle

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.
    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0,r=0.0, v=0.0,rho=1014,g=9.80665,L=68,U_norm=8,xg=-3.38):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.r = r
        self.rho = rho
        self.g = g
        self.L = L
        self.U_norm = U_norm
        self.xg = xg
        self.m = 634.9*10**(-5) * (0.5*rho*L**3)
        self.Izz = 2.63*10**(-5) * (0.5*rho*L**5)


    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.
        Stanley Control uses bicycle model.
        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt  #u x t
        self.y += self.v * np.sin(self.yaw) * dt  #v x t
        self.yaw += - self.v / L * np.tan(delta) * dt #psi
        #self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt

        u1 = self.v * np.cos(self.yaw)  # surge vel.
        v1 = self.v * np.sin(self.yaw)  # sway vel.
        r1 = self.v / L * np.tan(delta)  # yaw vel.

        U = np.sqrt(u1 ** 2 + v1 ** 2)  # speed

        # X - Coefficients
        Xu_dot = -31.0323 * 10 ** (-5) * (0.5 * self.rho * self.L ** 3)
        Xuu = -167.7891 * 10 ** (-5) * (0.5 * self.rho * self.L ** 2)
        Xvr = 209.5232 * 10 ** (-5) * (0.5 * self.rho * self.L ** 3)
        Xdel = -2.382 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))
        Xdd = -242.1647 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))
        # N - Coefficients
        Nv_dot = 19.989 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Nr_dot = -29.956 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 5))
        Nuv = -164.080 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Nur = -175.104 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Nrr = -156.364 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 5))
        Nrv = -579.631 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Ndel = -166.365 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3) * (U ** 2))
        # Y - Coefficients
        Yv_dot = -700.907 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yr_dot = -52.018 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 4))
        Yuv = -1010.163 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2))
        Yur = 233.635 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yvv = -316.746 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2))
        Yvr = -1416.083 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Yrv = -324.593 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 3))
        Ydel = 370.6 * 10 ** (-5) * (0.5 * self.rho * (self.L ** 2) * (U ** 2))

        # Hydrodynamic
        Ta = -Xuu * self.U_norm * self.U_norm  # Assumption: constant
        Xhyd = Xuu * u1 * np.abs(u1) + Xvr * v1 * r1 + Ta
        Yhyd = Yuv * np.abs(u1) * v1 + Yur * u1 * r1 + Yvv * v1 * np.abs(v1) + Yvr * v1 * np.abs(r1) \
               + Yrv * r1 * np.abs(v1)
        Nhyd = Nuv * np.abs(u1) * v1 + Nur * np.abs(u1) * r1 + Nrr * r1 * np.abs(r1) \
               + Nrv * r1 * np.abs(v1)

        Xrudder = 0  # Xdd * rudder_angle * rudder_angle + Xdel * rudder_angle
        Yrudder = Ydel * delta
        Nrudder = Ndel * delta

        H = np.array([[self.m - Xu_dot, 0, 0, 0],
                      [0, self.m - Yv_dot, self.m * self.xg - Yr_dot, 0],
                      [0, self.m * self.xg - Nv_dot, self.Izz - Nr_dot, 0],
                      [0, 0, 0, 1]])
        f = np.array([Xhyd + self.m * (v1 * r1 + self.xg * r1 ** 2) + Xrudder,
                      Yhyd - self.m * u1 * r1 + Yrudder,
                      Nhyd - self.m * self.xg * u1 * r1 + Nrudder,
                      r1])
        # output = np.matmul(np.linalg.inv(H), f).reshape((4))
        output = np.linalg.solve(H, f)

        u = u1 + output[0] * dt
        print(U)
        v = v1 + output[1] * dt
        r = r1 + output[2] * 180. / math.pi * dt

        self.v = np.sqrt(u ** 2 + v ** 2)
        self.r = r


        # position[2] = position[2] + velocity[2] * dt
        #
        # if position[2] > 180:
        #     position[2] = position[2] - 360
        #
        # if position[2] < -180:
        #     position[2] = position[2] + 360

        # rot_matrix = np.array(
        #     [[math.cos(position[2] * math.pi / 180),
        #       -math.sin(position[2] * math.pi / 180)],
        #      [math.sin(position[2] * math.pi / 180),
        #       math.cos(position[2] * math.pi / 180)]])

        # pdot = np.array([[velocity[0]], [velocity[1]]])
        # pdot = pdot.reshape(2,1)
        # pdot = np.array([velocity[0], velocity[1]])
        # XYvel = np.dot(rot_matrix, pdot)
        # position[0] += XYvel[0] * dt
        # position[1] += XYvel[1] * dt



def pid_control(target, current):
    """
    Proportional control for the speed.
    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.
    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx


    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.
    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    ax = [900.0, 1000.0, 2000.0, 1000.0, 2000.0,1000.0,2000.0]
    ay = [160.0, 200.0, 400.0,600.0, 800.0,1000.0,1200.0]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    target_speed = 16 / 3.6  # [m/s]

    max_simulation_time = 200.0

    # Initial state
    state = State(x=800.0, y=150, yaw=np.radians(00.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)

    while max_simulation_time >= time and last_idx > target_idx:
        ai = pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
        #print(target_idx)
        # if target_idx > 1800:
        #     ai =
        state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()