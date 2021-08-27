# UAV Control with AI - Enhanced Linear Cascaded Control for Quadrotor via Neural Modelling

Trajectory generation and control algorithms using AI for UAVs with ROS wrapping.

This repository provides the code associated to the paper : Enhanced Linear Cascaded Control for Quadrotor via Neural Modelling
[Paper link]
[Video link]

This project uses external software such as Mavros, PX4 or Gazebo. Below are the links directing to their documentations:

[PX4 Development Guide](https://dev.px4.io/v1.9.0/en/)

[PX4-Firmware](https://github.com/PX4/Firmware)

[Mavros](https://github.com/mavlink/mavros/)

[Sitl-Gazebo](https://github.com/PX4/sitl_gazebo)

## Installation
For the installation, you need to have ROS melodic (or kinetic) installed, a catkin workspace and Gazebo. Follow the online documentation to set up your environement.

[ROS Installation](http://wiki.ros.org/melodic/Installation/Ubuntu)

[Catkin Workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace)

[Gazebo](http://gazebosim.org/tutorials?tut=install_ubuntu&cat=install)

### Prerequisites
Install mavros

```bash
sudo apt install ros-melodic-mavros ros-melodic-mavros-extras
```

Mavros request the GeographicLib datasets, install it by running the install_geographiclib_datasets.sh script

```bash
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
sudo ./install_geographiclib_datasets.sh
```
Install libgstreamer

```bash
sudo apt install libgstreamer1.0-dev
```

Initialize rosdep and update it

```bash
sudo rosdep init
rosdep update
```

Clone sitl_gazebo and PX4 Firmware

```bash
cd ~/catkin_ws/src/
git clone --recursive https://github.com/PX4/sitl_gazebo
```
```bash
git clone --recursive https://github.com/PX4/Firmware px4
```
or to replicate paper results, we recommand you to use a specific version of px4 using the following line code :
```bash
git clone --recursive -b kakutef7_ekf_v1.11.3 https://github.com/gipsa-lab-uav/PX4-Autopilot.git
```

**Note:** If you have troubles installing the different packages, it is recommended to read the related documentation.

### Install uav_control_ai package
Clone the uav_control_ai repository
```bash
cd ~/catkin_ws/src/
git clone https://github.com/gipsa-lab-uav/uav_control_ai
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

To finish the installation run:
```bash
cd ..
catkin_make
```

Then source your setup.bash:

```bash
source devel/setup.bash
```

### Testing the installation
In a terminal, run the following line:
```bash
roslaunch uav_control_ai example.launch
```
Three windows must appear :
- A gazebo window with the iris
- A window displaying live information sush as the thrust, position, etc.,
- A window with the desired generated trajectory.

If everything works fine,the iris quadcopter should takeoff and start the trajectory.

**Note** : QGroundControl can be open in parallel to check if everything is interfacing correctly.

## How to use it ?

### Simulation

1. [Trajectory] Design the desired trajectory using 'trajectory_gen.py' file. Examples and explanations are available in the file comments.

2. [Controller] Use 'simu_params.yaml' to specify :
- mass and hover compensation coefficient,
- control gains on the position loop and attitude loop,
- select 'clin' as controller type for linear control (proportional derivative).

3. [Simulation] Run:
```bash
roslaunch uav_control_ai example.launch
```
4. [Data] Data are stored in 'StateCommandData.msg' available in rosbag folder.

5. [Learning] To learn error dynamics you can use learning_error_dynamics node:
```bash
rosrun uav_control_ai learning_error_dynamics.py
```
At the end of the process, you can save the DNN model in nodes/dnn_model.

6. [Simulation with AI] Modify the 'simu_params.yaml' to specify :
- 'clin_ai' to use the DNN Enhanced Linear Cascaded controller_rates
- The DNN selected model path

Then run again :
```bash
roslaunch uav_control_ai example.launch
```
