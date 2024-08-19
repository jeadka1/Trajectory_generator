import numpy as np
import matplotlib.pyplot as plt


class MobileRobotTrajectoryGenerator:
    def __init__(self, waypoints, max_speeds, max_acceleration, total_time):
        self.waypoints = np.array(waypoints)  # Waypoints as a numpy array
        self.max_speeds = np.array(max_speeds)  # Max speed at each waypoint
        self.max_acceleration = max_acceleration  # Maximum acceleration for the robot
        self.total_time = total_time  # Total time to complete the trajectory
        self.n_points = len(waypoints)
        self.segment_times = np.linspace(0, total_time, self.n_points)
        self.velocities = []  # Store calculated velocities

    def compute_distances(self):
        """Compute distances between consecutive waypoints."""
        distances = []
        for i in range(self.n_points - 1):
            dist = np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i])
            distances.append(dist)
        return np.array(distances)

    def compute_curvatures(self):
        """Compute curvatures based on waypoints."""
        curvatures = []
        for i in range(1, self.n_points - 1):
            a = self.waypoints[i - 1]
            b = self.waypoints[i]
            c = self.waypoints[i + 1]
            # Calculate curvature based on three consecutive points
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ac = np.linalg.norm(c - a)
            s = (ab + bc + ac) / 2
            area = np.sqrt(s * (s - ab) * (s - bc) * (s - ac))
            curvature = 4 * area / (ab * bc * ac) if ab * bc * ac != 0 else 0
            curvatures.append(curvature)
        curvatures.insert(0, 0)  # First point has no curvature
        curvatures.append(0)  # Last point has no curvature
        return np.array(curvatures)

    def generate_trajectory(self):
        """ Generate trajectory with speed and acceleration adjustments."""
        distances = self.compute_distances()
        curvatures = self.compute_curvatures()

        self.velocities = []  # Clear velocities for re-calculation

        for i in range(self.n_points - 1):
            # Adjust speed based on curvature and max speed
            curvature_factor = max(1.0, curvatures[i] * 10)  # Adjust as needed
            max_speed = min(self.max_speeds[i], self.max_speeds[i] / curvature_factor)

            # Check if maximum speed is achievable given the segment distance
            distance = distances[i]
            achievable_speed = np.sqrt(self.max_acceleration * distance)  # Max achievable speed in the segment

            # Limit the speed to the achievable speed if the segment is too short
            final_speed = min(max_speed, achievable_speed)
            self.velocities.append(final_speed)

    def trapezoidal_velocity_profile(self, segment_idx, t, segment_duration):
        """Compute position, velocity, and acceleration using a trapezoidal velocity profile."""
        start_point = self.waypoints[segment_idx]
        end_point = self.waypoints[segment_idx + 1]
        distance = np.linalg.norm(end_point - start_point)

        # Compute times for acceleration and deceleration phases
        max_speed = self.velocities[segment_idx]
        accel_time = max_speed / self.max_acceleration
        decel_time = max_speed / self.max_acceleration
        cruise_time = segment_duration - (accel_time + decel_time)

        if cruise_time < 0:  # If there's no cruise phase (triangle profile)
            accel_time = np.sqrt(2 * distance / self.max_acceleration)
            decel_time = accel_time
            cruise_time = 0

        # Calculate position by integrating velocity over time
        if t < accel_time:  # Acceleration phase
            velocity = self.max_acceleration * t
            position = 0.5 * self.max_acceleration * t ** 2  # Integrating velocity during acceleration
            acceleration = self.max_acceleration
        elif t < accel_time + cruise_time:  # Constant speed phase
            velocity = max_speed
            position = 0.5 * self.max_acceleration * accel_time ** 2 + max_speed * (t - accel_time)  # Constant speed
            acceleration = 0
        else:  # Deceleration phase
            decel_time_elapsed = t - (accel_time + cruise_time)
            velocity = max_speed - self.max_acceleration * decel_time_elapsed
            position = (0.5 * self.max_acceleration * accel_time ** 2 +
                        max_speed * cruise_time +
                        max_speed * decel_time_elapsed - 0.5 * self.max_acceleration * decel_time_elapsed ** 2)
            acceleration = -self.max_acceleration

        # Ensure that the position doesn't exceed the segment distance
        position = min(position, distance)

        # Interpolate position between start and end points based on the distance traveled
        position_ratio = position / distance if distance != 0 else 0
        interpolated_position = (1 - position_ratio) * start_point + position_ratio * end_point

        return interpolated_position, velocity, acceleration

    def get_position(self, t):
        """Get position at time t."""
        if t > self.total_time:
            t = self.total_time
        elif t < 0:
            t = 0

        segment_idx = min(int(np.floor(t / self.total_time * (self.n_points - 1))), self.n_points - 2)
        segment_start_time = self.segment_times[segment_idx]
        segment_end_time = self.segment_times[segment_idx + 1]
        segment_duration = segment_end_time - segment_start_time

        local_t = t - segment_start_time

        position, _, _ = self.trapezoidal_velocity_profile(segment_idx, local_t, segment_duration)
        return position

    def get_velocity(self, t):
        """Get velocity at time t."""
        if t > self.total_time:
            t = self.total_time
        elif t < 0:
            t = 0

        segment_idx = min(int(np.floor(t / self.total_time * (self.n_points - 1))), self.n_points - 2)
        segment_start_time = self.segment_times[segment_idx]
        segment_end_time = self.segment_times[segment_idx + 1]
        segment_duration = segment_end_time - segment_start_time

        local_t = t - segment_start_time

        _, velocity, _ = self.trapezoidal_velocity_profile(segment_idx, local_t, segment_duration)
        return velocity

    def get_acceleration(self, t):
        """Get acceleration at time t."""
        if t > self.total_time:
            t = self.total_time
        elif t < 0:
            t = 0

        segment_idx = min(int(np.floor(t / self.total_time * (self.n_points - 1))), self.n_points - 2)
        segment_start_time = self.segment_times[segment_idx]
        segment_end_time = self.segment_times[segment_idx + 1]
        segment_duration = segment_end_time - segment_start_time

        local_t = t - segment_start_time

        _, _, acceleration = self.trapezoidal_velocity_profile(segment_idx, local_t, segment_duration)
        return acceleration


# Example Usage:
waypoints = [
    [0, 0],  # Start point
    [2, 1],  # Waypoint 1
    [4, 0],  # Waypoint 2
    [6, -1],  # Waypoint 3
    [8, 0]  # End point
]

max_speeds = [1.0, 0.8, 0.6, 0.8, 1.0]  # Max speeds at each waypoint
max_acceleration = 0.5  # Maximum acceleration (m/s^2)
total_time = 50  # Total time to complete the trajectory

# Create the trajectory generator
trajectory_gen = MobileRobotTrajectoryGenerator(waypoints, max_speeds, max_acceleration, total_time)

# Generate the trajectory
trajectory_gen.generate_trajectory()

# Simulate the trajectory
times = np.linspace(0, total_time, 100)
positions = []
velocities = []
accelerations = []

for t in times:
    pos = trajectory_gen.get_position(t)
    vel = trajectory_gen.get_velocity(t)
    accl = trajectory_gen.get_acceleration(t)
    positions.append(pos)
    velocities.append(vel)
    accelerations.append(accl)
    print(f"Time {t:.2f} s: Position {pos}, Velocity {vel} m/s, Acceleration {accl} m/s²")

# Convert positions, velocities, and accelerations to numpy arrays for easier indexing
positions = np.array(positions)
velocities = np.array(velocities)
accelerations = np.array(accelerations)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot positions
plt.subplot(3, 1, 1)
plt.plot(positions[:, 0], positions[:, 1], label='Trajectory')
plt.scatter(np.array(waypoints)[:, 0], np.array(waypoints)[:, 1], color='red', label='Waypoints')
plt.title('Robot Trajectory with Speed and Acceleration Adjustments')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()

# Plot velocities
plt.subplot(3, 1, 2)
plt.plot(times, velocities, label='Speed')
plt.title('Robot Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()

# Plot accelerations
plt.subplot(3, 1, 3)
plt.plot(times, accelerations, label='Acceleration')
plt.title('Robot Acceleration over Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

plt.tight_layout()
plt.show()
