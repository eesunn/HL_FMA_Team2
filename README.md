# HL_FMA_Team2

---

## 개발 환경

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS2 | Humble Hawksbill |
| Python | 3.10 |

---

## 세팅 순서

### 1. ROS2 Humble 설치

```bash
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu \
  $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
```

### 2. ROS2 추가 패키지 설치

```bash
sudo apt install -y \
  ros-humble-rplidar-ros \
  ros-humble-slam-toolbox \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-robot-localization \
  ros-humble-robot-state-publisher \
  ros-humble-xacro \
  ros-humble-tf2-ros \
  ros-humble-cv-bridge \
  ros-humble-rviz2
```

### 3. 시스템 패키지 설치

```bash
sudo apt install -y \
  can-utils \
  python3-can \
  python3-serial
```

### 4. Python 라이브러리 설치

```bash
pip3 install \
  opencv-python==4.11.0.86 \
  numpy==1.26.4 \
  ultralytics==8.4.60 \
  torch==2.12.0 \
  torchvision==0.27.0 \
  pyserial==3.5 \
  python-can==3.3.2 \
  depthai==3.6.1 \
  scipy==1.8.0 \
  Pillow==9.0.1
```

### 5. 워크스페이스 Clone 및 빌드

```bash
git clone https://github.com/eesunn/HL_FMA_Team2.git
cd HL_FMA_Team2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

bashrc에 자동 source 추가:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "source ~/HL_FMA_Team2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
