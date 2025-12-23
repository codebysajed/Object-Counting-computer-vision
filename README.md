# Object Counting using YOLO and BOTSORT
This project counts the number of objects of a specific class (e.g. car, bus, truck) entering and exiting a region of interest (ROI) in a video. The project uses YOLO (You Only Look Once) for object detection and BOTSORT (Simple Online and Realtime Tracking with a Deep Association Metric) for object tracking.

## How to use
1. Clone the repository: `git clone https://github.com/codebysajed/Object-Counting-computer-vision.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the object counting script: `python main.py`

## Output
The script will output a video with the object count displayed on the screen. The video will have two lines, one for entering objects and one for exiting objects. The lines will be colored based on the class of the object.

## Configuration
The script can be configured by changing the following variables:
* `limitIn`: The coordinates of the line for entering objects.
* `limitOut`: The coordinates of the line for exiting objects.
* `class_name`: The names of the classes to be detected.
* `conf`: The confidence threshold for object detection.
* `tracker`: The tracker configuration file.

## Example
The following example will count the number of cars, buses, and trucks entering and exiting a region of interest in a video.

