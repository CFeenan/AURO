import sys

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from auro_interfaces.msg import StringWithPose
from assessment_interfaces.msg import  Zone, ZoneList, Item, ItemList
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
SCAN_FRONT = 0
SCAN_LEFT = 1
SCAN_BACK = 2
SCAN_RIGHT = 3


# Finite state machine (FSM) states
class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2
    LOOKING = 3
    RETRIEVING = 4 
    
    
class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        #self.declare_parameter('x', 0.0)
        #self.declare_parameter('y', 0.0)
        #self.declare_parameter('yaw', 0.0)
        
        # zoneDict = {}

        #self.initial_x = self.get_parameter('x').get_parameter_value().double_value
        #self.initial_y = self.get_parameter('y').get_parameter_value().double_value
        #self.initial_yaw = self.get_parameter('yaw').get_parameter_value().double_value

        # Class variables used to store persistent values between executions of callbacks and control loop
        self.state = State.FORWARD # Current FSM state
        self.pose = Pose() # Current pose (position and orientation), relative to the odom reference frame
        self.previous_pose = Pose() # Store a snapshot of the pose for comparison against future poses
        self.yaw = 0.0 # Angle the robot is facing (rotation around the Z axis, in radians), relative to the odom reference frame
        self.previous_yaw = 0.0 # Snapshot of the angle for comparison against future angles
        self.turn_angle = 0.0 # Relative angle to turn to in the TURNING state
        self.turn_direction = TURN_LEFT # Direction to turn in the TURNING state
        self.goal_distance = random.uniform(1.0, 2.0) # Goal distance to travel in FORWARD state
        self.scan_triggered = [False] * 4 # Boolean value for each of the 4 LiDAR sensor sectors. True if obstacle detected within SCAN_THRESHOLD
        self.items = ItemList()
        self.zones = ZoneList()
        self.look_count = 0
        self.front_left_ranges = []
        self.front_right_ranges = []
        self.carrying_item = False

        
        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value
        
        client_callback_group = MutuallyExclusiveCallbackGroup()
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        
        self.pick_up_service = self.create_client(ItemRequest, '/pick_up_item', callback_group=client_callback_group)
        self.offload_service = self.create_client(ItemRequest, '/offload_item', callback_group=client_callback_group)
        
        self.zone_subscriber = self.create_subscription(
            ZoneList,
            '/robot1/zone',
            self.zone_callback,
            10
        )
    
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/robot1/odom',
            self.odom_callback,
            10)
        
        self.item_subscriber = self.create_subscription(
            ItemList,
            '/robot1/items',
            self.item_callback,
            10,  
            callback_group=timer_callback_group
        )
        
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/robot1/scan',
            self.scan_callback,
            QoSPresetProfiles.SENSOR_DATA.value,
             callback_group=timer_callback_group
        )
        
        
        self.cmd_vel_publisher = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        
        self.marker_publisher = self.create_publisher(StringWithPose, '/robot1/marker_input', 10)
        
        # Creates a timer that calls the control_loop method repeatedly - each loop represents single iteration of the FSM
        self.timer_period = 0.1 # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg
        self.get_logger().debug(f"Received items: {self.items.data}")

        
    def zone_callback(self, msg):
        self.zones = msg
        
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
        # Group scan ranges into 4 segments
        # Front, left, and right segments are each 60 degrees

        front_ranges = msg.ranges[331:359] + msg.ranges[0:30]
        left_ranges  = msg.ranges[31:90] # 31 to 90 degrees (31 to 90 degrees)
        back_ranges  = msg.ranges[91:270] # 91 to 270 degrees (91 to -90 degrees)
        right_ranges = msg.ranges[271:330] # 271 to 330 degrees (-30 to -91 degrees)

        # Store True/False values for each sensor segment, based on whether the nearest detected obstacle is closer than SCAN_THRESHOLD
        self.scan_triggered[SCAN_FRONT] = min(front_ranges) < SCAN_THRESHOLD 
        self.scan_triggered[SCAN_LEFT]  = min(left_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_BACK]  = min(back_ranges)  < SCAN_THRESHOLD
        self.scan_triggered[SCAN_RIGHT] = min(right_ranges) < SCAN_THRESHOLD
        
          # Process ranges for specific segments
        self.front_left_ranges = msg.ranges[345:360] + msg.ranges[0:15]  # Adjust for circular indexing
        self.front_right_ranges = msg.ranges[15:30] + msg.ranges[330:345]
        
    
    # Control loop for the FSM - called periodically by self.timer
    def control_loop(self):

        # Send message to rviz_text_marker node
        marker_input = StringWithPose()
        marker_input.text = str(self.state) # Visualise robot state as an RViz marker
        marker_input.pose = self.pose # Set the pose of the RViz marker to track the robot's pose
        self.marker_publisher.publish(marker_input)

        self.get_logger().info(f"{self.state}")

        
        match self.state:
            
            case State.FORWARD:
              
                if self.look_count >= 4: 
                    self.state = State.LOOKING
                    print("now looking!")
                    return
                
                if len(self.zones.data) > 0 and self.carrying_item == True:
                    self.state = State.RETRIEVING
                    return

                
                # if self.scan_triggered[SCAN_FRONT]:
                #     # Check specific segments within SCAN_FRONT
                #     obstacle_front_left = any(
                #         r < SCAN_THRESHOLD for r in self.front_left_ranges if r < float('inf') and not math.isnan(r)
                #     )
                #     obstacle_front_right = any(
                #         r < SCAN_THRESHOLD for r in self.front_right_ranges if r < float('inf') and not math.isnan(r)
                #     )

                #     if obstacle_front_left and obstacle_front_right:
                #         self.turn_direction = TURN_RIGHT
                #         self.get_logger().info("Obstacles detected in both front-left and front-right sectors, turning right.")
                #     elif obstacle_front_left:
                #         self.turn_direction = TURN_RIGHT
                #         self.get_logger().info("Obstacle detected in front-left sector, turning right.")
                #     elif obstacle_front_right:
                #         self.turn_direction = TURN_LEFT
                #         self.get_logger().info("Obstacle detected in front-right sector, turning left.")
                #     else:
                #         self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                #         self.get_logger().info("General obstacle detected in front sector, randomly choosing turn direction.")

                #     self.previous_yaw = self.yaw
                #     self.state = State.TURNING
                #     self.turn_angle = random.uniform(50, 80)
                #     self.get_logger().info(f"Turning {('left' if self.turn_direction == TURN_LEFT else 'right')} by {self.turn_angle:.2f} degrees")
                #     return
                
                if self.scan_triggered[SCAN_FRONT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                    self.get_logger().info("Detected obstacle in front, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    self.look_count += 1
                    return
                
        
                if self.scan_triggered[SCAN_LEFT] or self.scan_triggered[SCAN_RIGHT]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = 45

                    if self.scan_triggered[SCAN_LEFT] and self.scan_triggered[SCAN_RIGHT]:
                        self.turn_direction = random.choice([TURN_LEFT, TURN_RIGHT])
                        self.get_logger().info("Detected obstacle to both the left and right, turning " + ("left" if self.turn_direction == TURN_LEFT else "right") + f" by {self.turn_angle:.2f} degrees")
                    elif self.scan_triggered[SCAN_LEFT]:
                        self.turn_direction = TURN_RIGHT
                        self.get_logger().info(f"Detected obstacle to the left, turning right by {self.turn_angle} degrees")
                    else: # self.scan_triggered[SCAN_RIGHT]
                        self.turn_direction = TURN_LEFT
                        self.get_logger().info(f"Detected obstacle to the right, turning left by {self.turn_angle} degrees")
                    return
                
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

                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
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
                
                if len(self.zones.data) > 0:
                    self.state = State.RETRIEVING
                    self.get_logger().info("Zone found during LOOKING, switching to RETRIEVING state.")
                    return
                    

                # Initialize LOOKING state parameters
                if self.cumulative_turn == 0.0:
                    self.initial_yaw = self.yaw
                    self.cumulative_turn = 0.0
                    self.turning_direction = TURN_LEFT  # You can randomize this if desired
                    self.get_logger().info("Entering LOOKING state, starting 360-degree rotation.")

                # Command rotation
                msg = Twist()
                msg.angular.z = self.turning_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                # Calculate yaw difference
                current_yaw = self.yaw
                yaw_diff = angles.normalize_angle(current_yaw - self.initial_yaw)
                yaw_diff_deg = math.degrees(yaw_diff)

                # Update cumulative_turn based on direction
                if self.turning_direction == TURN_LEFT:
                    self.cumulative_turn += math.degrees(yaw_diff)
                else:
                    self.cumulative_turn -= math.degrees(yaw_diff)

                # Normalize cumulative_turn
                self.cumulative_turn = angles.normalize_angle(math.radians(self.cumulative_turn))
                self.cumulative_turn_deg = math.degrees(math.fabs(self.cumulative_turn))

                self.get_logger().debug(f"LOOKING: Cumulative Turn = {self.cumulative_turn_deg:.2f} degrees")

                if self.cumulative_turn_deg >= 360.0:
                    # Completed full rotation without finding an item
                    self.cumulative_turn = 0.0
                    self.state = State.FORWARD
                    self.get_logger().info("Completed 360-degree rotation without finding items, switching to FORWARD state.")

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
                
                estimated_distance = 69.0 * float(item.diameter) ** -0.89
                
                self.get_logger().info(f'Estimated distance {estimated_distance}')
                                
                if estimated_distance <= 0.35:
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    
                    try:
                        future = self.pick_up_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item picked up.')
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
                
                zone = self.zones.data[0] 
                
                estimated_distance = 69.0 * float(zone.size) ** -0.89
                
                if estimated_distance <= 0.3:
                    rqt = ItemRequest.Request()
                    rqt.robot_id = self.robot_id
                    
                    try:
                        future = self.offload_service.call_async(rqt)
                        rclpy.spin_until_future_complete(self, future)
                        response = future.result()
                        if response.success:
                            self.get_logger().info('Item dropped off.')
                            self.state = State.LOOKING
                            self.zones.data = []
                        else:
                            self.get_logger().info('Unable to drop off item: ' + response.message)
                    except Exception as e:
                       self.get_logger().info('Exception' + e)

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
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()