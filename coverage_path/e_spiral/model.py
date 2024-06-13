import math

class ESpiralCPP:
    """
    Compute turning points for sprial traversal.
    Each turning point contains coordinates and turning angle
    Default direction is clockwise
    """

    def __init__(self, drone, altitude, cam_angle):
        self.altitude = altitude
        self.cam_angle = cam_angle
        self.cam_fov_h = drone.fov_h
        self.img_width = drone.img_width
        self.img_height = drone.img_height
        self.max_distance = drone.max_distance
        self.waypoints = []
        self.gcc_width = 2 * altitude
        self.gcc_height = int(self.gcc_width * drone.img_height / drone.img_width)
        self.offset_y = int(self.gcc_height / 2  + altitude * math.tan(
            math.radians(90 - abs(cam_angle) - drone.fov_v/2)))
    
    def calculate_waypoints(self):
        """
        Compute waypoints for spiral traversal
        From the current path, make a 90-degree clockwise turn.
        Check if the next waypoint is in visisted list
        If not, add to the waypoints list. 
        If yes, continue to the same path without turning. Check again.
        """
        wps = []
        start_wp = [-self.offset_y, 0, -self.altitude]
        wps.append(start_wp)
        distance = 0
        i = 0

        while distance < self.max_distance:
            wp = wps[-1]
            x, y = wp[0], wp[1]
            if i == 0: # 0 degree, right, attempt to move down
                next_x = x
                next_y = y - self.gcc_height
                distance += self.gcc_height
            elif i == 1: # -90 degree, down, attempt to move left
                next_x = x - self.gcc_width
                next_y = y
                distance += self.gcc_width
            elif i == 2: # -180 degree, left, attempt to move up
                next_x = x
                next_y = y + self.gcc_height
                distance += self.gcc_height
            elif i == 3: # -270 degree, up, attempt to move right
                next_x = x + self.gcc_width
                next_y = y
                distance += self.gcc_width
            next_wp = [next_y, next_x, -self.altitude]
            if distance > self.max_distance:
                break
            else:
                if next_wp not in wps:
                    wps.append(next_wp)
                    i = 0 if i == 3 else i+1
                else:
                    i = 3 if i == 0 else i-1

        self.waypoints = wps

        return wps

    def get_waypoints(self):
        if not self.waypoints:
            self.calculate_waypoints()
        return self.waypoints
    
    def calculate_wps_steps(self, step_length):
        wp_steps = []
        wps = self.get_waypoints()
        for i in range(len(wps)-1):
            wp1 = wps[i]
            wp2 = wps[i+1]
            wp_steps = wp_steps + self.calculate_steps(wp1, wp2, step_length)
        return wp_steps
    
    def calculate_steps(self, wp1, wp2, step_length):
        steps = []
        x1, y1, z1 = wp1
        x2, y2, z2 = wp2
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        num_steps = int(distance / step_length)
        steps = [[x1 + i * (x2 - x1) / num_steps,
                  y1 + i * (y2 - y1) / num_steps,
                  z1 + i * (z2 - z1) / num_steps] for i in range(num_steps)]
        steps.append(wp2)
        return steps
    
    def set_altitude(self, altitude):
        self.altitude = altitude

    def set_cam_angle(self, cam_angle):
        self.cam_angle = cam_angle