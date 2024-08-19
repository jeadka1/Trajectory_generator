import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math

class JointTrajectoryInterpolationComponent:
    def __init__(self, waypoints, start_vel, end_vel):
        self._readyPos = None
        self._isTargetReached = False
        self._isTrajSet = False
        self._Waypoint = []
        self.nPts = 0
        self.nSplines = 0
        self.dim = 2
        self._t0 = 0
        self._segTime = 10.0
        self._s = 0.0
        self.a0 = None
        self.a1 = None
        self.a2 = None
        self.a3 = None
        self.start_vel = start_vel
        self.end_vel = end_vel
        self.curvatures = []

    def setPath(self, path):
        """ Set the waypoints for interpolation. """
        self._readyPos = path[0]
        self._Waypoint = path
        self.nPts = len(path)
        self.nSplines = self.nPts - 1
        self.dim = len(path[0])  # Assuming 3-dimensional vectors for position (x, y, theta)

        # Initialize coefficients for cubic spline
        self.a0 = np.zeros((self.dim, self.nSplines))
        self.a1 = np.zeros((self.dim, self.nSplines))
        self.a2 = np.zeros((self.dim, self.nSplines))
        self.a3 = np.zeros((self.dim, self.nSplines))

        # Call Interpolation
        self.interpolate()
        self.calculate_curvatures()
        print("Set Path")

    def interpolate(self):
        for j in range(self.dim):
            coeff_A = np.zeros((4 * self.nSplines, 4 * self.nSplines))
            coeff_b = np.zeros((4 * self.nSplines, 1))

            for i in range(self.nSplines):
                # N conditions: x = f_i(s_i)
                coeff_A[i, i * 4 + 0] = i ** 3
                coeff_A[i, i * 4 + 1] = i ** 2
                coeff_A[i, i * 4 + 2] = i
                coeff_A[i, i * 4 + 3] = 1.0
                coeff_b[i, 0] = self._Waypoint[i][j]

                # N conditions: x = f_i(s_(i+1))
                coeff_A[self.nSplines + i, i * 4 + 0] = (i + 1) ** 3
                coeff_A[self.nSplines + i, i * 4 + 1] = (i + 1) ** 2
                coeff_A[self.nSplines + i, i * 4 + 2] = i + 1
                coeff_A[self.nSplines + i, i * 4 + 3] = 1.0
                coeff_b[self.nSplines + i, 0] = self._Waypoint[i + 1][j]

                # 2*(N-1) conditions: f_i'(s_(i+1)) = f_(i+1)'(s_(i+1)), f_i''(s_(i+1)) = f_(i+1)''(s_(i+1))
                if i < self.nSplines - 1:
                    coeff_A[2 * self.nSplines + i, i * 4 + 0] = 3.0 * (i + 1) ** 2
                    coeff_A[2 * self.nSplines + i, i * 4 + 1] = 2.0 * (i + 1)
                    coeff_A[2 * self.nSplines + i, i * 4 + 2] = 1.0
                    coeff_A[2 * self.nSplines + i, (i + 1) * 4 + 0] = -3.0 * (i + 1) ** 2
                    coeff_A[2 * self.nSplines + i, (i + 1) * 4 + 1] = -2.0 * (i + 1)
                    coeff_A[2 * self.nSplines + i, (i + 1) * 4 + 2] = -1.0
                    coeff_b[2 * self.nSplines + i, 0] = 0.0

                    coeff_A[3 * self.nSplines - 1 + i, i * 4 + 0] = 6.0 * (i + 1)
                    coeff_A[3 * self.nSplines - 1 + i, i * 4 + 1] = 2.0
                    coeff_A[3 * self.nSplines - 1 + i, (i + 1) * 4 + 0] = -6.0 * (i + 1)
                    coeff_A[3 * self.nSplines - 1 + i, (i + 1) * 4 + 1] = -2.0
                    coeff_b[3 * self.nSplines - 1 + i, 0] = 0.0

            # Adding start velocity constraint
            coeff_A[4 * self.nSplines - 2, 0 * 4 + 0] = 3.0 * 0 ** 2
            coeff_A[4 * self.nSplines - 2, 0 * 4 + 1] = 2.0 * 0
            coeff_A[4 * self.nSplines - 2, 0 * 4 + 2] = 1.0
            coeff_b[4 * self.nSplines - 2, 0] = self.start_vel[j]

            # Adding end velocity constraint
            coeff_A[4 * self.nSplines - 1, (self.nSplines - 1) * 4 + 0] = 3.0 * self.nSplines ** 2
            coeff_A[4 * self.nSplines - 1, (self.nSplines - 1) * 4 + 1] = 2.0 * self.nSplines
            coeff_A[4 * self.nSplines - 1, (self.nSplines - 1) * 4 + 2] = 1.0
            coeff_b[4 * self.nSplines - 1, 0] = self.end_vel[j]
            # 2 conditions: f_0''(s_0) = 0, f_n''(s_(n+1)) = 0
            # coeff_A[4 * self.nSplines - 2, 0 * 4 + 0] = 6.0 * 0
            # coeff_A[4 * self.nSplines - 2, 0 * 4 + 1] = 2.0
            # coeff_b[4 * self.nSplines - 2, 0] = 0.0
            #
            # coeff_A[4 * self.nSplines - 1, (self.nSplines - 1) * 4 + 0] = 6.0 * self.nSplines
            # coeff_A[4 * self.nSplines - 1, (self.nSplines - 1) * 4 + 1] = 2.0
            # coeff_b[4 * self.nSplines - 1, 0] = 0.0

            # Solve the system for the coefficients
            coeff = np.linalg.solve(coeff_A, coeff_b)

            # Extract the coefficients for each spline segment
            for i in range(self.nSplines):
                self.a0[j, i] = coeff[4 * i + 0]
                self.a1[j, i] = coeff[4 * i + 1]
                self.a2[j, i] = coeff[4 * i + 2]
                self.a3[j, i] = coeff[4 * i + 3]

    def calculate_curvatures(self):
        """ Calculate curvatures between waypoints. """
        for i in range(self.nSplines):
            x1, y1 = self._Waypoint[i][:2]
            x2, y2 = self._Waypoint[i + 1][:2]
            dx = x2 - x1
            dy = y2 - y1
            curvature = abs(dx * y2 - dy * x2) / (dx ** 2 + dy ** 2) ** 1.5 if dx ** 2 + dy ** 2 != 0 else 0
            self.curvatures.append(curvature)
        print("Curvatures:", self.curvatures)

    def setTraj(self, t0):
        self._t0 = t0
        self._segTime = 10.0
        self._isTargetReached = False
        self._isTrajSet = True

    def traj(self, time):
        """ Generate trajectory given time. """
        posDes = np.zeros(self.dim)
        velDes = np.zeros(self.dim)
        accDes = np.zeros(self.dim)

        if time <= self._t0:
            return self._readyPos, velDes, accDes

        # Compute the normalized time step
        self._s = (time - self._t0) / self._segTime
        if self._s > self.nSplines:
            self._s = self.nSplines

        idx = min(int(np.floor(self._s)), self.nSplines - 1)

        sdot = 1.0 / self._segTime
        sddot = 0.0  # Assuming constant velocity

        # Apply curvature-based velocity reduction
        curvature_factor = max(1.0, self.curvatures[idx] * 10)  # Tune factor as needed
        sdot /= curvature_factor  # Reduce velocity by curvature factor

        for i in range(self.dim):
            # Compute position, velocity, and acceleration based on the cubic polynomial
            posDes[i] = (self.a0[i, idx] * self._s ** 3 +
                         self.a1[i, idx] * self._s ** 2 +
                         self.a2[i, idx] * self._s +
                         self.a3[i, idx])

            velDes[i] = (3.0 * self.a0[i, idx] * self._s ** 2 +
                         2.0 * self.a1[i, idx] * self._s +
                         self.a2[i, idx]) * sdot

            accDes[i] = (6.0 * self.a0[i, idx] * self._s +
                         2.0 * self.a1[i, idx]) * sdot ** 2

        if self._t0 + self.nSplines * self._segTime <= time:
            self._readyPos = posDes
            self._isTrajSet = False
            self._isTargetReached = True
            self._s = 0.0

        return posDes, velDes, accDes

    def isTargetReached(self):
        return self._isTargetReached


# Example Usage:
# Given g_x and g_y values
g_x = [0, 0]
g_y = [0, 1]

# Create waypoints list using a for loop
waypoints = []
for i in range(len(g_x)):
    distance = math.sqrt(g_x[i] ** 2 + g_y[i] ** 2)
    angle = math.atan2(g_y[i], g_x[i])
    waypoints.append([distance, angle])

# Print the waypoints
for i, waypoint in enumerate(waypoints):
    print(f"Waypoint {i}: Distance = {waypoint[0]}, Angle = {waypoint[1]}")
start_velocity = [0, 0]  # Start velocities in each dimension (x, y, theta)
end_velocity = [0.2, 0.2]  # End velocities in each dimension (x, y, theta)

trajectory_component = JointTrajectoryInterpolationComponent(waypoints, start_velocity, end_velocity)
trajectory_component.setPath(waypoints)
trajectory_component.setTraj(0)

# Simulate over time
positions = []
velocities = []
accelerations = []
times = np.linspace(0, 10, num=20)
for t in times:
    pos, vel, acc = trajectory_component.traj(t)
    print(f"Time {t:.2f}: Position {pos}, Velocity {vel}, Acceleration {acc}")
    positions.append(pos)
    velocities.append(vel)
    accelerations.append(acc)

# Convert lists to numpy arrays for easier plotting
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)

# Plotting
plt.figure(figsize=(14, 8))

# Plot positions
plt.subplot(3, 1, 1)
for i in range(positions.shape[1]):
    plt.plot(times, positions[:, i], label=f'Joint {i + 1}')
plt.title('Joint Positions')
plt.ylabel('Position (degrees)')
plt.legend()

# Plot velocities
plt.subplot(3, 1, 2)
for i in range(velocities.shape[1]):
    plt.plot(times, velocities[:, i], label=f'Joint {i + 1}')
plt.title('Joint Velocities')
plt.ylabel('Velocity (degrees/s)')
plt.legend()

# Plot accelerations
plt.subplot(3, 1, 3)
for i in range(accelerations.shape[1]):
    plt.plot(times, accelerations[:, i], label=f'Joint {i + 1}')
plt.title('Joint Accelerations')
plt.ylabel('Acceleration (degrees/s^2)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.show()
