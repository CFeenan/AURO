import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Float32, Int8MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from auro_interfaces.msg import StringWithPose
from assessment_interfaces.msg import  Zone, ZoneList, Item, ItemList, RobotList, Robot
from auro_interfaces.srv import ItemRequest


from tf_transformations import euler_from_quaternion
import angles

from enum import Enum
import random
import math

LINEAR_VELOCITY  = 0.3 # Metres per second
ANGULAR_VELOCITY = 0.5 # Radians per second

TURN_LEFT = 1 # Postive angular velocity turns left
TURN_RIGHT = -1 # Negative angular velocity turns right

SCAN_THRESHOLD = 0.5 # Metres per second

 # Array indexes for sensor sectors
SCAN_FRONT_A = 0
SCAN_FRONT_B = 1

SCAN_LEFT_A = 2
SCAN_LEFT_B = 3

SCAN_RIGHT_A = 4
SCAN_RIGHT_B = 5

SCAN_BACK = 6

class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    LOOKING = 3
    RETRIEVING = 4 
    WAITING = 5
    
class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.FORWARD # Current FSM state
        self.prior_state = State.FORWARD
        self.pose = Pose() # Current pose (position and orientation), relative to the odom reference frame
        self.previous_pose = Pose() # Store a snapshot of the pose for comparison against future poses
        self.yaw = 0.0 # Angle the robot is facing (rotation around the Z axis, in radians), relative to the odom reference frame
        self.previous_yaw = 0.0 # Snapshot of the angle for comparison against future angles
        self.turn_angle = 0.0 # Relative angle to turn to in the TURNING state
        self.turn_direction = TURN_LEFT # Direction to turn in the TURNING state
        self.goal_distance = random.uniform(1.0, 2.0) # Goal distance to travel in FORWARD state
        self.scan_triggered = [False] * 7 # Boolean value for each of the 7 LiDAR sensor sectors. True if obstacle detected within SCAN_THRESHOLD
        self.items = ItemList()
        self.zones = ZoneList()
        self.robots = RobotList()
        self.look_count = 0
        self.front_left_ranges = []
        self.front_right_ranges = []
        self.carrying_item = False
        self.current_item_colour = ""
        self.looking_initialised = False
        self.retrieving_procedure = False 
        self.wait_count = False
        self.zone_log = [0,0,0,0]
        self.colours = {"RED": 1, "GREEN": 2, "BLUE": 3}
        
        # Track cumulative rotation in LOOKING state
        self.cumulative_turn = 0.0

        

        self.robot_id = self.get_namespace().strip("/")
        
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        
        self.pick_up_service = self.create_client(ItemRequest, '/pick_up_item', callback_group=client_callback_group)
        self.offload_service = self.create_client(ItemRequest, '/offload_item', callback_group=client_callback_group)
        
        self.zone_subscriber = self.create_subscription(
            ZoneList,
            f"/{self.robot_id}/zone",
            self.zone_callback,
            10
        )
        
        self.zone_subscriber = self.create_subscription(
            Int8MultiArray,
            "/zone_log",
            self.zone_log_callback,
            10
        )
    
        self.odom_subscriber = self.create_subscription(
            Odometry,
            f"/{self.robot_id}/odom",
            self.odom_callback,
            10)
        
        self.robot_subscriber = self.create_subscription(
            RobotList,
            f"/{self.robot_id}/robots",
            self.robot_callback,
            10)
        
        self.item_subscriber = self.create_subscription(
            ItemList,
            f"/{self.robot_id}/items",
            self.item_callback,
            10,  
            callback_group=timer_callback_group
        )
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            f"/{self.robot_id}/scan",
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
             callback_group=timer_callback_group
        )
        
        self.cmd_vel_publisher = self.create_publisher(Twist, f"/{self.robot_id}/cmd_vel", 10)
        
        self.marker_publisher = self.create_publisher(StringWithPose, f"/{self.robot_id}/marker_input", 10)
        
        self.zone_log_publisher = self.create_publisher(Int8MultiArray, "/zone_log", 10)
        
        # Creates a timer that calls the control_loop method repeatedly - each loop represents single iteration of the FSM
        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg
        self.get_logger().debug(f"Received items: {self.items.data}")

    def zone_callback(self, msg):
        self.zones = msg
        
    def zone_log_callback(self, msg):
        self.zone_log = msg.data
        
    def robot_callback(self, msg):
        self.robots = msg
        
    def odom_callback(self, msg):
        self.pose = msg.pose.pose # Store the pose in a class variable

        # Uses tf_transformations package to convert orientation from quaternion to Euler angles (RPY = roll, pitch, yaw)
        # https://github.com/DLu/tf_transformations
        #
        # Roll (rotation around X axis) and pitch (rotation around Y axis) are discarded
        (roll, pitch, yaw) = euler_from_quaternion([self.pose.orientation.x,
                                                    self.pose.orientation.y,
                                                    self.pose.orientation.z,
                                                    self.pose.orientation.w])
        
        orientation = Float32()
        orientation.data = yaw
        normalised = angles.normalize_angle(yaw - self.yaw)
        self.get_logger().info(f"Publishing diff yaw normalised: {normalised:.3f}")
        self.yaw = yaw # Store the yaw in a class variable
        #self.orientation_publisher.publish(orientation)
        
    def scan_callback(self, msg):
        # Group scan ranges into 7 segments
        # Front, left, and right segments are each 60 degrees

        front_range_A = msg.ranges[0:30]
        front_range_B = msg.ranges[331:359]
        right_range_A = msg.ranges[271:300] 
        right_range_B = msg.ranges[301:330]
        left_range_A = msg.ranges[31:60]
        left_range_B = msg.ranges[61:90]
        back_ranges  = msg.ranges[91:270] # 91 to 270 degrees (91 to -90 degrees)

        # Store True/False values for each sensor segment, based on whether the nearest detected obstacle is closer than SCAN_THRESHOLD
        self.scan_triggered[SCAN_FRONT_A] = min(front_range_A) < SCAN_THRESHOLD 
        self.scan_triggered[SCAN_FRONT_B] = min(front_range_B) < SCAN_THRESHOLD

        self.scan_triggered[SCAN_LEFT_A] = min(left_range_A) < SCAN_THRESHOLD   
        self.scan_triggered[SCAN_LEFT_B] = min(left_range_B) < SCAN_THRESHOLD
        
        self.scan_triggered[SCAN_RIGHT_A] = min(right_range_A) < SCAN_THRESHOLD   
        self.scan_triggered[SCAN_RIGHT_B] = min(right_range_B) < SCAN_THRESHOLD

        self.scan_triggered[SCAN_BACK]  = min(back_ranges)  < SCAN_THRESHOLD
    
    
    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):

        # Send message to rviz_text_marker node
        marker_input = StringWithPose()
        marker_input.text = str(self.state) # Visualise robot state as an RViz marker
        marker_input.pose = self.pose # Set the pose of the RViz marker to track the robot's pose
        self.marker_publisher.publish(marker_input)

        self.get_logger().info(f"{self.state}")
        
        if len(self.robots.data) >= 1:
            
            close_robot = any(robot.size > 0.25 for robot in self.robots.data)
            self.get_logger().info(f"{self.robots.data[0].size}")
            if close_robot == True and self.state != State.WAITING:
                self.prior_state = self.state
                self.state = State.WAITING
            elif close_robot ==True:
                return
            else:
                if self.state == State.WAITING:
                    self.state = self.prior_state
        else:
            if self.state == State.WAITING:
                self.state = self.prior_state
                    
                
        match self.state:
            
            case State.FORWARD:
                
                if self.look_count >= 4: 
                    self.state = State.LOOKING
                    print("now looking!")
                    self.look_count = 0
                    return
                
                if len(self.zones.data) > 0 and self.carrying_item == True:
                    distance_check = any(float(zone.size) > 0.05 for zone in self.zones.data)
                    return
                
                
                if self.scan_triggered[SCAN_FRONT_A] or self.scan_triggered[SCAN_FRONT_B]:
                    
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.look_count += 1 
                    
                    if self.scan_triggered[SCAN_FRONT_A] and not self.scan_triggered[SCAN_FRONT_B]:

                        self.turn_direction = TURN_RIGHT
                        self.turn_angle = 50
                        self.get_logger().info("detect in front A , turning " + f"{self.turn_direction}" + f" by {self.turn_angle:.2f} degrees")
                        return

                    if self.scan_triggered[SCAN_FRONT_B] and not self.scan_triggered[SCAN_FRONT_A]:
                        self.turn_direction = TURN_LEFT
                        self.turn_angle = 50
                        self.get_logger().info("detect in front B, turning " + f"{self.turn_direction}" + f" by {self.turn_angle:.2f} degrees")
                        return
                    
                    else:
                        self.turn_direction = TURN_RIGHT
                        self.turn_angle = 180
                        return
                    
                if self.scan_triggered[SCAN_LEFT_A] or self.scan_triggered[SCAN_LEFT_B]:
                    self.turn_direction = TURN_RIGHT
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.look_count +=1 
                    
                    if self.scan_triggered[SCAN_LEFT_A]:
                        self.turn_angle = 35
                        
                    if self.scan_triggered[SCAN_LEFT_B]:
                        self.turn_angle = 25
                    self.get_logger().info("detect in LEFT A OR B, turning " + f"{self.turn_direction}" + f" by {self.turn_angle:.2f} degrees")
                    
                if self.scan_triggered[SCAN_RIGHT_A] or self.scan_triggered[SCAN_RIGHT_B]:
                    self.turn_direction = TURN_LEFT
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.look_count +=1 
                    
                    if self.scan_triggered[SCAN_RIGHT_A]:
                        self.turn_angle = 35
                        
                    if self.scan_triggered[SCAN_RIGHT_B]:
                        self.turn_angle = 25
                        
                    self.get_logger().info("detect in LEFT A OR B, turning " + f"{self.turn_direction}" + f" by {self.turn_angle:.2f} degrees")
                    
                if len(self.items.data) > 0 and self.carrying_item == False:
                    self.state = State.COLLECTING
                    return
               
                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.get_logger().info(f"Publishing cmd_vel: linear.x={msg.linear.x}, angular.z={msg.angular.z}")
                self.cmd_vel_publisher.publish(msg)
                
                difference_x = self.pose.position.x - self.previous_pose.position.x
                difference_y = self.pose.position.y - self.previous_pose.position.y
                distance_travelled = math.sqrt(difference_x ** 2 + difference_y ** 2)

                # self.get_logger().info(f"Driven {distance_travelled:.2f} out of {self.goal_distance:.2f} metres")

                if distance_travelled >= self.goal_distance:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(30, 150)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Goal reached, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")

            case State.TURNING:

                if len(self.items.data) > 0 and self.carrying_item == False:
                    self.state = State.COLLECTING
                    return
                
                if len(self.zones.data) > 0 and self.carrying_item == True:
                    
                    distance_check = any(float(zone.size) > 0.04 for zone in self.zones.data)
                    
                    if distance_check == True:
                        self.state = State.RETRIEVING
                        return

                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # self.get_logger().info(f"Turned {math.degrees(math.fabs(yaw_difference)):.2f} out of {self.turn_angle:.2f} degrees")

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)                

                if math.fabs(yaw_difference) >= math.radians(self.turn_angle):
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")
                    
              
            case State.LOOKING:

                if len(self.items.data) > 0 and self.carrying_item == False:
                    self.state = State.COLLECTING
                    self.get_logger().info("Item found during LOOKING, switching to COLLECTING state.")
                    return
                
                if len(self.zones.data) > 0 and self.carrying_item == True:
                    distance_check = any(float(zone.size) > 0.038 for zone in self.zones.data)
                    
                    if distance_check == True:
                        self.state = State.RETRIEVING
                        self.get_logger().info("Zone found during LOOKING, switching to RETRIEVING state.")
                        return
                        

                if self.looking_initialised == False:
                    # Send stop command
                    stop_msg = Twist()
                    stop_msg.linear.x = 0.0
                    stop_msg.angular.z = 0.0
                    self.cmd_vel_publisher.publish(stop_msg)
                    self.get_logger().info("Stopping movement before rotation.")

                    # Initialize rotation
                    self.previous_yaw = self.yaw
                    self.turn_angle = 360
                    self.turn_direction = TURN_LEFT
                    self.get_logger().info("Entering LOOKING state, starting 360-degree rotation.")
                    self.looking_initialised = True
                    self.cumulative_turn = 0.0

                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)
                
                yaw_diff = angles.normalize_angle(self.yaw - self.previous_yaw)
                if self.turn_direction == TURN_LEFT:
                    self.cumulative_turn += math.degrees(yaw_diff)
                else:
                    self.cumulative_turn -= math.degrees(yaw_diff)

                self.previous_yaw = self.yaw
                self.get_logger().info(f"Cumulative rotation: {self.cumulative_turn:.2f}°")

                # Completed 360°, return to FORWARD
                if abs(self.cumulative_turn) >= 360.0:
                    self.cumulative_turn = 0.0
                    self.looking_initialised = False
                    self.get_logger().info("Completed 360° LOOK, returning to FORWARD.")
                    self.state = State.FORWARD

                


            case State.COLLECTING:

                if len(self.items.data) == 0:
                    self.previous_pose = self.pose
                    self.state = State.FORWARD
                    return

                closestItem = 0
                
                for index, itemData in enumerate(self.items.data):
                    if itemData.diameter >= self.items.data[closestItem].diameter:
                        closestItem = index
                     
                        
                item = self.items.data[closestItem] 
                self.get_logger().info('diameter' + str(item.diameter))
                
                estimated_distance = 69.0 * float(item.diameter) ** -0.89
                
                self.get_logger().info(f'Estimated distance {estimated_distance}')
                                
                if estimated_distance <= 0.35:
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        self.executor.spin_until_future_complete(future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item picked up.')
                            self.carrying_item = True
                            self.current_item_colour = item.colour
                            self.state = State.RETRIEVING
                            self.items.data = []
                    
                            
                        else:
                            self.get_logger().info('Unable to pick up item: ' + response.message)
                    except Exception as e:
                       self.get_logger().info('Exception' + e)
                    

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                msg.angular.z = item.x / 320.0
                self.cmd_vel_publisher.publish(msg)
                
            case State.RETRIEVING:
                
                if len(self.zones.data) == 0:
                    self.previous_pose = self.pose
                    self.state = State.LOOKING
                    return
                
                closestZone = 0 
                
                for index, zoneData in enumerate(self.zones.data):
                    if zoneData.size >= self.zones.data[closestZone].size:
                        closestZone = index
                
                zone = self.zones.data[closestZone]
                self.get_logger().info('Zoneinfo' + str(zone))
                
                associated_colour_number = self.colours[self.current_item_colour]
                if self.zone_log[zone.zone-1] != associated_colour_number and self.zone_log[zone.zone-1] != 0:
                    self.state = State.LOOKING
                    return
                        
                if zone.size <= 0.04:
                    self.state = State.LOOKING
                    return
                
                # Improved speed calculation
                k = 0.015  # Scaling factor
                max_speed = 0.7  # Maximum speed in m/s
                min_speed = 0.3  # Minimum speed in m/s

                try:
                    zone_size = float(zone.size)
                    if zone_size <= 0:
                        zone_size = 0.05  # Prevent division by zero or negative sizes
                    estimated_speed = k / zone_size
                    estimated_speed = max(min_speed, min(estimated_speed, max_speed))
                except Exception as e:
                    self.get_logger().error(f"Error calculating speed: {e}")
                    estimated_speed = min_speed
                    
                if float(zone.size) >= 1.00:
                    
                    while self.wait_count <= 5:
                        self.wait_count += 1
                        return
                    
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    
                    try:
                        future = self.offload_service.call_async(rqt)
                        self.executor.spin_until_future_complete(future)
                        response = future.result()
                        if response.success:
                            
                            msg = Twist()
                            msg.linear.x = 0.0
                            self.cmd_vel_publisher.publish(msg)
                            
                            zone_msg = Int8MultiArray()
                            associated_colour_number = self.colours[self.current_item_colour]
                            self.zone_log[zone.zone-1] = associated_colour_number
                            
                            zone_msg.data = self.zone_log
                            self.zone_log_publisher.publish(zone_msg)
                            
                            self.get_logger().info('Item dropped off.')
                            self.state = State.LOOKING
                            self.carrying_item = False
                            self.zones.data = []
                            
                        else:
                            self.get_logger().info('Unable to drop off item: ' + response.message)
                    except Exception as e:
                       self.get_logger().info('Exception' + e)
                
                msg = Twist()
                msg.linear.x = estimated_speed
                msg.angular.z = zone.x / 320.0
                self.cmd_vel_publisher.publish(msg)
            
            case State.WAITING:
                msg = Twist()
                msg.linear.x = 0.0
                self.cmd_vel_publisher.publish(msg)
                
            case _:
                pass
            
    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()
    
def main(args=None):

    rclpy.init(args = args, signal_handler_options = SignalHandlerOptions.NO)

    node = RobotController()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()