#include <ros/ros.h>

#include <actionlib/client/simple_action_client.h>
#include <tabletop_object_detector/TabletopDetection.h>
#include <tabletop_collision_map_processing/TabletopCollisionMapProcessing.h>
#include <object_manipulation_msgs/PickupAction.h>
#include <object_manipulation_msgs/PlaceAction.h>
#include <object_manipulator/tools/mechanism_interface.h>

#include <arm_navigation_msgs/CollisionObject.h>
#include <arm_navigation_msgs/MoveArmAction.h>
#include <arm_navigation_msgs/utils.h>

#include <iostream>


/**
  This is demo to detect transparent objects in clutter on a right side of the table and move them to the left part.
  The code is based on the ROS pick and place tutorial:
  http://www.ros.org/wiki/pr2_tabletop_manipulation_apps/Tutorials/Writing%20a%20Simple%20Pick%20and%20Place%20Application

  Note 1. that tabletop_collision_map_processing was needed to be patched to process cases
  when the detector returns only models without clusters as in our case.
  Note 2. The demo is not very stable so use carefully.
*/


//the function from pr2_interactive_manipulation
inline std::vector<double> getSidePosition(std::string arm_name)
{
  std::vector<double> sp(7,0.0);
  if (arm_name=="right_arm")
  {
    //sp[0] = -2.135; sp[1] = 0.803; sp[2] = -1.732; sp[3] = -1.905; sp[4] = -2.369; sp[5] = -1.680; sp[6] = 1.398;
    //sp[0] = -2.135; sp[1] = -0.18; sp[2] = -1.732; sp[3] = -1.905; sp[4] = -2.369; sp[5] = -1.680; sp[6] = 1.398;
    sp[0] = -2.135; sp[1] = -0.02; sp[2] = -1.64; sp[3] = -2.07; sp[4] = -1.64; sp[5] = -1.680; sp[6] = 1.398;
  }
  else
  {
    //sp[0] = 2.135; sp[1] = 0.803; sp[2] = 1.732; sp[3] = -1.905; sp[4] = 2.369; sp[5] = -1.680; sp[6] = 1.398;
    //sp[0] = 2.135; sp[1] = -0.18; sp[2] = 1.732; sp[3] = -1.905; sp[4] = 2.369; sp[5] = -1.680; sp[6] = 1.398;
    sp[0] = 2.135; sp[1] = -0.02; sp[2] = 1.64; sp[3] = -2.07; sp[4] = 1.64; sp[5] = -1.680; sp[6] = 1.398;
  }
  return sp;
}


int main(int argc, char **argv)
{
  //TODO: process command line parameters
  bool resetCollisions = (argc == 1);

  //initialize the ROS node
  ros::init(argc, argv, "pick_and_place_app");
  ros::NodeHandle nh;

  //set service and action names
  const std::string OBJECT_DETECTION_SERVICE_NAME =
    "/object_recognition_translated";
  const std::string COLLISION_PROCESSING_SERVICE_NAME =
    "/tabletop_collision_map_processing/tabletop_collision_map_processing";
  const std::string PICKUP_ACTION_NAME =
    "/object_manipulator/object_manipulator_pickup";
  const std::string PLACE_ACTION_NAME =
    "/object_manipulator/object_manipulator_place";

  std::string arms[] = {"left_arm", "right_arm"};
  const int armsCount = 2;

  //create service and action clients
  ros::ServiceClient object_detection_srv;
  ros::ServiceClient collision_processing_srv;
  actionlib::SimpleActionClient<object_manipulation_msgs::PickupAction>
    pickup_client(PICKUP_ACTION_NAME, true);
  actionlib::SimpleActionClient<object_manipulation_msgs::PlaceAction>
    place_client(PLACE_ACTION_NAME, true);

  //wait for detection client
  while ( !ros::service::waitForService(OBJECT_DETECTION_SERVICE_NAME,
        ros::Duration(2.0)) && nh.ok() )
  {
    ROS_INFO("Waiting for object detection service to come up");
  }
  if (!nh.ok()) exit(0);
  object_detection_srv =
    nh.serviceClient<tabletop_object_detector::TabletopDetection>
    (OBJECT_DETECTION_SERVICE_NAME, true);

  //wait for collision map processing client
  while ( !ros::service::waitForService(COLLISION_PROCESSING_SERVICE_NAME,
        ros::Duration(2.0)) && nh.ok() )
  {
    ROS_INFO("Waiting for collision processing service to come up");
  }
  if (!nh.ok()) exit(0);
  collision_processing_srv =
    nh.serviceClient
    <tabletop_collision_map_processing::TabletopCollisionMapProcessing>
    (COLLISION_PROCESSING_SERVICE_NAME, true);

  //wait for pickup client
  while(!pickup_client.waitForServer(ros::Duration(2.0)) && nh.ok())
  {
    ROS_INFO_STREAM("Waiting for action client " << PICKUP_ACTION_NAME);
  }
  if (!nh.ok()) exit(0);

  //wait for place client
  while(!place_client.waitForServer(ros::Duration(2.0)) && nh.ok())
  {
    ROS_INFO_STREAM("Waiting for action client " << PLACE_ACTION_NAME);
  }
  if (!nh.ok()) exit(0);


  if (resetCollisions)
  {
    //call collision map processing
    ROS_INFO("Calling collision map processing");
    tabletop_collision_map_processing::TabletopCollisionMapProcessing
      processing_call;
    //ask for the existing map and collision models to be reset
    processing_call.request.reset_collision_models = true;
    processing_call.request.reset_attached_models = true;
    if (!collision_processing_srv.call(processing_call))
    {
      ROS_ERROR("Collision map processing service failed");
      return -1;
    }
  }

  int demoIteration = 0;
  geometry_msgs::Point previousPlacing;
  geometry_msgs::Pose previousDetectedPose;
  geometry_msgs::Pose firstDetectedPose;
  previousDetectedPose.position.x = 0.0;
  previousDetectedPose.position.y = 0.0;
  previousDetectedPose.position.z = 0.0;
  while(true)
  {
    //call the tabletop detection
    ROS_INFO("Calling tabletop detector");
    tabletop_object_detector::TabletopDetection detection_call;
    //we want recognized database objects returned
    //we don't want the individual object point clouds returned
    detection_call.request.return_clusters = false;
    //set this to false if you are using the pipeline without the database
    detection_call.request.return_models = true;
    detection_call.request.num_models = 1;
    if (!object_detection_srv.call(detection_call))
    {
      ROS_ERROR("Tabletop detection service failed");
      return -1;
    }
    if (detection_call.response.detection.result !=
        detection_call.response.detection.SUCCESS)
    {
      ROS_ERROR("Tabletop detection returned error code %d",
          detection_call.response.detection.result);
      return -1;
    }
    if (detection_call.response.detection.clusters.empty() &&
        detection_call.response.detection.models.empty() )
    {
      ROS_ERROR("The tabletop detector detected the table, "
          "but found no objects");
      return -1;
    }


    //call collision map processing
    ROS_INFO("Calling collision map processing");
    tabletop_collision_map_processing::TabletopCollisionMapProcessing
      processing_call;
    //pass the result of the tabletop detection
    processing_call.request.detection_result =
      detection_call.response.detection;
    //don't reset recognized models and collision objects
    processing_call.request.reset_collision_models = false;
    processing_call.request.reset_attached_models = true;
    //ask for the results to be returned in base link frame
    processing_call.request.desired_frame = "base_link";
    //processing_call.request.desired_frame = "/head_mount_kinect_rgb_optical_frame";
    if (!collision_processing_srv.call(processing_call))
    {
      ROS_ERROR("Collision map processing service failed");
      return -1;
    }

    //the collision map processor returns instances of graspable objects
    if (processing_call.response.graspable_objects.empty())
    {
      ROS_ERROR("Collision map processing returned no graspable objects");
      return -1;
    }

    geometry_msgs::Pose detectedObjectPose = processing_call.response.graspable_objects[0].potential_models[0].pose.pose;
    std::cout << "detected pose: " << processing_call.response.graspable_objects[0].potential_models[0].pose << std::endl;
    if (demoIteration == 0)
    {
      firstDetectedPose = detectedObjectPose;
    }


    int armIndex = 0;
    geometry_msgs::Vector3Stamped direction;
    object_manipulation_msgs::PickupResult pickup_result;
    for (; armIndex < armsCount; ++armIndex)
    {
      //call object pickup
      ROS_INFO("Calling the pickup action");
      //pass one of the graspable objects returned
      //by the collision map processor
      object_manipulation_msgs::PickupGoal pickup_goal;
      pickup_goal.target = processing_call.response.graspable_objects.at(0);
      //pass the name that the object has in the collision environment
      //this name was also returned by the collision map processor
      pickup_goal.collision_object_name =
        processing_call.response.collision_object_names.at(0);
      //pass the collision name of the table, also returned by the collision
      //map processor
      pickup_goal.collision_support_surface_name =
        processing_call.response.collision_support_surface_name;
      //pick up the object with an arm
      pickup_goal.arm_name = arms[armIndex];
      //we will be lifting the object along the "vertical" direction
      //which is along the z axis in the base_link frame
      direction.header.stamp = ros::Time::now();
      direction.header.frame_id = "base_link";
      direction.vector.x = 0;
      direction.vector.y = 0;
      direction.vector.z = 1;
      pickup_goal.lift.direction = direction;
      //request a vertical lift of 10cm after grasping the object
      //  pickup_goal.lift.desired_distance = 0.1;
      pickup_goal.lift.desired_distance = 0.1;
      pickup_goal.lift.min_distance = 0.05;
      //do not use tactile-based grasping or tactile-based lift
      pickup_goal.use_reactive_lift = false;
      pickup_goal.use_reactive_execution = false;
      //send the goal
      pickup_client.sendGoal(pickup_goal);
      while (!pickup_client.waitForResult(ros::Duration(10.0)))
      {
        ROS_INFO("Waiting for the pickup action...");
      }
      pickup_result =
        *(pickup_client.getResult());
      if (pickup_client.getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
      {
        ROS_ERROR("The pickup action has failed with result code %d",
            pickup_result.manipulation_result.value);
        if (armIndex == 0)
        {
          continue;
        }
        else
        {
          return -1;
        }
      }
      else
      {
        break;
      }
    }


    //put the object down
    ROS_INFO("Calling the place action");
    object_manipulation_msgs::PlaceGoal place_goal;

    //create a place location, offset by 10 cm from the pickup location
    const float eps = 1e-4;

    //put object on a right side of the table
    const float xshift = (firstDetectedPose.position.x - detectedObjectPose.position.x);
    const float xmin = xshift;
    const float xmax = xshift + 0.6;
    const float xstep = 0.01;

    const float yshift = (demoIteration == 0) ? -0.6 : (previousDetectedPose.position.y - detectedObjectPose.position.y) + previousPlacing.y;
    const float ymin = yshift;
    const float ymax = firstDetectedPose.position.y + 0.05;
    const float ystep = 0.01;

    const float dz = 0.04;
    for (float dy = ymin; dy < ymax + eps; dy += ystep)
    {
      for (float dx = xmin; dx < xmax + eps; dx += xstep)
      {
        geometry_msgs::PoseStamped place_location;
        place_location.header.frame_id = processing_call.response.graspable_objects.at(0).reference_frame_id;
        //identity pose
        place_location.pose.orientation.w = 1;
        place_location.header.stamp = ros::Time::now();
        place_location.pose.position.x += dx;
        place_location.pose.position.y += dy;
        place_location.pose.position.z += dz;
        //place at the prepared location
        place_goal.place_locations.push_back(place_location);
      }
    }

    //the collision names of both the objects and the table
    //same as in the pickup action
    place_goal.collision_object_name =
      processing_call.response.collision_object_names.at(0);
    place_goal.collision_support_surface_name =
      processing_call.response.collision_support_surface_name;
    //information about which grasp was executed on the object,
    //returned by the pickup action
    place_goal.grasp = pickup_result.grasp;
    //use the arm to place
    place_goal.arm_name = arms[armIndex];
    //padding used when determining if the requested place location
    //would bring the object in collision with the environment
    place_goal.place_padding = 0.01;
    //how much the gripper should retreat after placing the object
    place_goal.desired_retreat_distance = 0.1;
    place_goal.min_retreat_distance = 0.05;
    //we will be putting down the object along the "vertical" direction
    //which is along the z axis in the base_link frame
    direction.header.stamp = ros::Time::now();
    direction.header.frame_id = "base_link";
    direction.vector.x = 0;
    direction.vector.y = 0;
    direction.vector.z = -1;
    place_goal.approach.direction = direction;
    //request a vertical put down motion of 10cm before placing the object
    place_goal.approach.desired_distance = 0.1;
    place_goal.approach.min_distance = 0.05;
    //we are not using tactile based placing
    place_goal.use_reactive_place = false;
    place_goal.allow_gripper_support_collision = false;
    //send the goal
    place_client.sendGoal(place_goal);
    while (!place_client.waitForResult(ros::Duration(10.0)))
    {
      ROS_INFO("Waiting for the place action...");
    }
    object_manipulation_msgs::PlaceResult place_result =
      *(place_client.getResult());

    std::cout << "tried locations:" << std::endl;
    for (size_t i = 0; i < place_result.attempted_locations.size(); ++i)
    {
      std::cout << place_result.attempted_locations[i] << std::endl;
    }

    if (place_client.getState() !=
        actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_ERROR("Place failed with error code %d",
          place_result.manipulation_result.value);

      ROS_ERROR("Cannot place the object automacaly, please move the arm manually and provide input when this is done");
      std::string str;
      std::cin >> str;
    }
    else
    {
      previousPlacing = place_result.place_location.pose.position;
      std::cout << "Placed to the position: " << previousPlacing.x << " " << previousPlacing.y << std::endl;
    }

#if 0 //move arm
    actionlib::SimpleActionClient<arm_navigation_msgs::MoveArmAction> move_arm("move_right_arm",true);
    move_arm.waitForServer();
    ROS_INFO("Connected to server");
    arm_navigation_msgs::MoveArmGoal goalA;

    goalA.motion_plan_request.group_name = "right_arm";
    goalA.motion_plan_request.num_planning_attempts = 1;
    goalA.motion_plan_request.planner_id = std::string("");
    goalA.planner_service_name = std::string("ompl_planning/plan_kinematic_path");
    //  goalA.motion_plan_request.allowed_planning_time = ros::Duration(5.0);
    goalA.motion_plan_request.allowed_planning_time = ros::Duration(20.0);

    //motion_planning_msgs::SimplePoseConstraint desired_pose;
    arm_navigation_msgs::SimplePoseConstraint desired_pose;
    desired_pose.header.frame_id = "torso_lift_link";
    desired_pose.link_name = "r_wrist_roll_link";
    desired_pose.pose.position.x = 0.75;
    desired_pose.pose.position.y = -0.188;
    desired_pose.pose.position.z = 0;

    desired_pose.pose.orientation.x = 0.0;
    desired_pose.pose.orientation.y = 0.0;
    desired_pose.pose.orientation.z = 0.0;
    desired_pose.pose.orientation.w = 1.0;

    desired_pose.absolute_position_tolerance.x = 0.02;
    desired_pose.absolute_position_tolerance.y = 0.02;
    desired_pose.absolute_position_tolerance.z = 0.02;

    desired_pose.absolute_roll_tolerance = 0.04;
    desired_pose.absolute_pitch_tolerance = 0.04;
    desired_pose.absolute_yaw_tolerance = 0.04;

    arm_navigation_msgs::addGoalConstraintToMoveArmGoal(desired_pose,goalA);

    if (nh.ok())
    {
      bool finished_within_time = false;
      move_arm.sendGoal(goalA);
      finished_within_time = move_arm.waitForResult(ros::Duration(200.0));
      if (!finished_within_time)
      {
        move_arm.cancelGoal();
        ROS_INFO("Timed out achieving goal A");
      }
      else
      {
        actionlib::SimpleClientGoalState state = move_arm.getState();
        bool success = (state == actionlib::SimpleClientGoalState::SUCCEEDED);
        if(success)
          ROS_INFO("Action finished: %s",state.toString().c_str());
        else
          ROS_INFO("Action failed: %s",state.toString().c_str());
      }
    }

#endif

    //move the arm to the side
    actionlib::SimpleActionClient<arm_navigation_msgs::MoveArmAction> move_arm("move_" + arms[armIndex],true);

    move_arm.waitForServer();
    ROS_INFO("Connected to server to move the arm");

    arm_navigation_msgs::MoveArmGoal goalB;
    std::vector<std::string> names(7);
    names[0] = "_shoulder_pan_joint";
    names[1] = "_shoulder_lift_joint";
    names[2] = "_upper_arm_roll_joint";
    names[3] = "_elbow_flex_joint";
    names[4] = "_forearm_roll_joint";
    names[5] = "_wrist_flex_joint";
    names[6] = "_wrist_roll_joint";
    for (size_t i = 0; i < names.size(); ++i)
    {
      names[i] = arms[armIndex][0] + names[i];
    }

    goalB.motion_plan_request.group_name = arms[armIndex];
    goalB.motion_plan_request.num_planning_attempts = 1;
    goalB.motion_plan_request.allowed_planning_time = ros::Duration(5.0);

    goalB.motion_plan_request.planner_id= std::string("");
    goalB.planner_service_name = std::string("ompl_planning/plan_kinematic_path");
    goalB.motion_plan_request.goal_constraints.joint_constraints.resize(names.size());

    for (unsigned int i = 0 ; i < goalB.motion_plan_request.goal_constraints.joint_constraints.size(); ++i)
    {
      goalB.motion_plan_request.goal_constraints.joint_constraints[i].joint_name = names[i];
      goalB.motion_plan_request.goal_constraints.joint_constraints[i].position = 0.0;
      goalB.motion_plan_request.goal_constraints.joint_constraints[i].tolerance_below = 0.1;
      goalB.motion_plan_request.goal_constraints.joint_constraints[i].tolerance_above = 0.1;
    }

    std::vector<double> sp = getSidePosition(arms[armIndex]);
    for (size_t i = 0; i < sp.size(); ++i)
    {
      goalB.motion_plan_request.goal_constraints.joint_constraints[i].position = sp[i];
    }

    if (nh.ok())
    {
      bool finished_within_time = false;
      move_arm.sendGoal(goalB);
      finished_within_time = move_arm.waitForResult(ros::Duration(120.0));
      if (!finished_within_time)
      {
        move_arm.cancelGoal();
        ROS_INFO("Timed out achieving goal A");
        ROS_ERROR("Cannot move the arm to a side, please use rviz to do this and provide the input when done");
        std::string str;
        std::cin >> str;
      }
      else
      {
        actionlib::SimpleClientGoalState state = move_arm.getState();
        bool success = (state == actionlib::SimpleClientGoalState::SUCCEEDED);
        if(success)
        {
          ROS_INFO("Action finished: %s",state.toString().c_str());
        }
        else
        {
          ROS_INFO("Action failed: %s",state.toString().c_str());
          ROS_ERROR("Cannot move the arm to a side, please use rviz to do this and provide the input when done");
          std::string str;
          std::cin >> str;
        }
      }
    }

#if 0 //mech_interface to move the arm to the side
    object_manipulator::MechanismInterface mech_interface;
    //  std::string arm_name = "right_arm";
    //  std::string arm_name = "r_arm_controller";
    std::string arm_name = "/r_arm_controller";
    arm_navigation_msgs::OrderedCollisionOperations empty_col;
    std::vector<arm_navigation_msgs::LinkPadding> empty_pad;
    if ( !mech_interface.attemptMoveArmToGoal(arm_name, getSidePosition(arm_name), empty_col, empty_pad) )
      std::cout << "Failed to move arm to side\n";
    else
      std::cout << "Moved arm to side\n";
#endif

    //success!
    ROS_INFO("Success! Object moved.");
    ++demoIteration;
    previousDetectedPose = detectedObjectPose;
  }
  return 0;
}

