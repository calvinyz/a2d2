"""
E-Spiral Coverage Path Planning
Computes spiral traversal waypoints for drone coverage with camera angle consideration.
"""

import math
from typing import List, Tuple, Optional
from drone.airsim_drone import AirsimDrone
from alert.alert_manager import AlertManager, AlertPriority, AlertType
from datetime import datetime

class ESpiralCPP:
    """
    Compute turning points for spiral traversal.
    
    Generates waypoints for a clockwise spiral pattern considering drone camera
    parameters and altitude. Each waypoint contains coordinates and turning angle.
    """

    def __init__(self, drone: AirsimDrone, altitude: float, cam_angle: float):
        """
        Initialize spiral path planner.
        
        Args:
            drone: Drone configuration object
            altitude: Flight altitude in meters
            cam_angle: Camera angle in degrees
        """
        self.drone = drone
        self.altitude = altitude
        self.cam_angle = cam_angle
        self.cam_fov_h = drone.fov_h
        self.max_distance = drone.max_distance
        
        # Image parameters
        self.img_width = drone.img_width
        self.img_height = drone.img_height
        
        # Ground coverage calculations
        self.gcc_width = 2 * altitude
        self.gcc_height = int(self.gcc_width * drone.img_height / drone.img_width)
        
        # Calculate vertical offset based on camera angle
        self.offset_y = int(
            self.gcc_height / 2 + altitude * math.tan(
                math.radians(90 - abs(cam_angle) - drone.fov_v/2)
            )
        )
        
        # Initialize waypoints list
        self.waypoints: List[List[float]] = []
    
    def calculate_waypoints(self) -> List[List[float]]:
        """
        Compute waypoints for spiral traversal.
        
        Generates waypoints by making 90-degree clockwise turns and checking
        if next point has been visited. If visited, continues on same path.
        
        Returns:
            list: List of [y, x, z] waypoint coordinates
        """
        try:
            wps = []
            start_wp = [-self.offset_y, 0, -self.altitude]
            wps.append(start_wp)
            distance = 0
            direction = 0  # 0: right, 1: down, 2: left, 3: up

            while distance < self.max_distance:
                current_wp = wps[-1]
                x, y = current_wp[0], current_wp[1]
                
                # Calculate next waypoint based on direction
                next_x, next_y, step_distance = self._get_next_position(x, y, direction)
                next_wp = [next_y, next_x, -self.altitude]
                
                distance += step_distance
                if distance > self.max_distance:
                    break
                    
                if next_wp not in wps:
                    wps.append(next_wp)
                    direction = (direction + 1) % 4  # Turn clockwise
                else:
                    direction = (direction - 1) % 4  # Continue straight

            self.waypoints = wps
            
            # Record waypoint generation in monitoring history
            if hasattr(self.drone, 'monitor'):
                self.drone.monitor.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "waypoint_generation",
                    "num_waypoints": len(wps),
                    "total_distance": distance
                })
            
            return wps
            
        except Exception as e:
            if hasattr(self.drone, 'monitor'):
                self.drone.monitor.alert_manager.send_alert(
                    f"Waypoint calculation failed: {str(e)}",
                    AlertPriority.HIGH
                )
            raise

    def _get_next_position(
        self,
        x: float,
        y: float,
        direction: int
    ) -> Tuple[float, float, float]:
        """
        Calculate next position based on current direction.
        
        Args:
            x: Current x coordinate
            y: Current y coordinate
            direction: Current direction (0-3)
            
        Returns:
            tuple: (next_x, next_y, step_distance)
        """
        if direction == 0:  # Right to down
            return x, y - self.gcc_height, self.gcc_height
        elif direction == 1:  # Down to left
            return x - self.gcc_width, y, self.gcc_width
        elif direction == 2:  # Left to up
            return x, y + self.gcc_height, self.gcc_height
        else:  # Up to right
            return x + self.gcc_width, y, self.gcc_width

    def get_waypoints(self) -> List[List[float]]:
        """Get computed waypoints, calculating if not already done."""
        if not self.waypoints:
            self.calculate_waypoints()
        return self.waypoints
    
    def calculate_wps_steps(self, step_length: float) -> List[List[float]]:
        """
        Calculate intermediate waypoints between turning points.
        
        Args:
            step_length: Distance between intermediate points
            
        Returns:
            list: List of intermediate waypoints
        """
        wp_steps = []
        wps = self.get_waypoints()
        
        for i in range(len(wps)-1):
            wp_steps.extend(self.calculate_steps(wps[i], wps[i+1], step_length))
            
        return wp_steps
    
    def calculate_steps(
        self,
        wp1: List[float],
        wp2: List[float],
        step_length: float
    ) -> List[List[float]]:
        """
        Calculate intermediate points between two waypoints.
        
        Args:
            wp1: Starting waypoint coordinates
            wp2: Ending waypoint coordinates
            step_length: Distance between steps
            
        Returns:
            list: List of intermediate waypoints
        """
        x1, y1, z1 = wp1
        x2, y2, z2 = wp2
        
        # Calculate distances and number of steps
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        num_steps = int(distance / step_length)
        
        # Generate intermediate points
        steps = [
            [
                x1 + i * dx / num_steps,
                y1 + i * dy / num_steps,
                z1 + i * dz / num_steps
            ] for i in range(num_steps)
        ]
        steps.append(wp2)
        
        return steps
    
    def set_altitude(self, altitude: float):
        """Update flight altitude."""
        self.altitude = altitude

    def set_cam_angle(self, cam_angle: float):
        """Update camera angle."""
        self.cam_angle = cam_angle

    def execute_path(self, step_length: float = 1.0) -> bool:
        """Execute computed path with drone."""
        try:
            waypoints = self.get_waypoints()
            
            # Record path execution start
            if hasattr(self.drone, 'monitor'):
                self.drone.monitor.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "path_execution_start",
                    "num_waypoints": len(waypoints)
                })
            
            for wp in waypoints:
                # Check drone status
                status = self.drone.get_status()
                if status["status"] == "ERROR":
                    raise RuntimeError("Drone in error state")
                    
                # Move to waypoint
                if not self.drone.move_to(*wp):
                    return False
                    
                # Check battery
                if status["battery"] < 20:
                    if hasattr(self.drone, 'monitor'):
                        self.drone.monitor.alert_manager.send_alert(
                            "Aborting path execution: Low battery",
                            AlertPriority.CRITICAL
                        )
                    return False
            
            return True
            
        except Exception as e:
            if hasattr(self.drone, 'monitor'):
                self.drone.monitor.alert_manager.send_alert(
                    f"Path execution failed: {str(e)}",
                    AlertPriority.HIGH
                )
            raise