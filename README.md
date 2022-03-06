# Introduction
## summarising objectives
- create semi-automated video processing analysis (reducing number of operators)
- processes normal match streams
- create framework for football data science research (academic or business)
- free and open source (will take additional work to avoid non compatible libraries with GPL3)


The project's aims are to collect track and event data from football footage. The ultimate goal is to be able to process any kind of footage but, for now, it will only process broadcast football matches.


## problems solved to achieve objectives
- general
    + low quality footage
    + weather and light visual conditions
    + detect refs by colour
    + detect people outside of the pitch
    + process different video/match segments
    + short video segments interrupt data collection
    + loads of moving parts
- detect humans
    + filter out referee(s)
    + detect players
    + identify players numbers
    + detects large human (noise, needs to be removed)
    + determine players position
    + playersbcrossing eachother
- detect ball
    + determine its position
    + track its position
    + 3D ball positioning estimation using physics model
    + ball tracking is suspended whenever an object obstructs the camera view
- detect pitch, by machine learning trained by creating image dataset
    + pitch geometry reconstruction
    + optical distortion
    + detect corner flags
    + detect goals
    + machine learning algorithm to
    + algorithm to determine inner section
    + different camera/editing/perspective
- map players and ball to pitch
    + predict possible position of off sight players
    + positioning flipping
- recognise players
    + player's numbers extraction
- action detection
- collect data
    + collect event data
    + collect tracking data
    + synchronise event and tracking data
    + synchronise footage and data timestamps

### pseudocode

pseudocode for image recognition

development of 3-d modelling
pseudocode for 3-d modelling data set generation
machine learning model and algorithm
geometry reconstruction algorithm


### images

![full text is here!!!!! description](./images/first.png)
players are detected but ball is not (purple means that object recognition just ran)
multiple players are detected in the same bounding box
refs are detected as well
one steward is also detected
streaming is stopped because ball is not found

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-32-51.png)
players bounding boxes in green means that it is tracking
manually labelling ball to continue stream

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-35-39.png)
ball tracking  is lost and tracks the numbers on the players back

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-36-05.png)
player tracking continues
new players appear on the screen but they are not detected until 30frame period runs object detection again

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-38-04.png)
ball tracking is lost again because of the pitch lines and player boots
some players previously detected are lost because of the backgroup from ads or pitch (not enough constract)

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-36-47.png)
ball needs to be labelled again to be tracked again

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-05-49.png)
the object detection is ran
all the human trackers are removed but the ball tracker
the ball tracker is not reset if the ball is still being tracked

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-06-03.png)
the players tracker had to be reset again from the <prev img>
ball tracker continues to run, regardless

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-06-18.png)
the ball tracker is wrong again, by tracking the player's back number

![full text is here!!!!! description](./images/Screenshot_2022-03-03_21-39-53.png)
this is a video segment imposed by the director
data cannot be collected
this is a short moment

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-06-43.png)
ball is tracked from the <prev img> since the label needs to be labelled for the stream to continue
there is a bug, a human is recognised due to noise

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-08-00.png)
players are detected again
the bug stil persists
a fan is recognised in the crowd

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-09-09.png)
description

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-09-45.png)
description

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-10-58.png)
description

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-12-29.png)
description

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-13-18.png)
description

![full text is here!!!!! description](./images/Screenshot_2022-03-03_23-19-11.png)
description


![text description](./images/coordinates.png)
description

![text description](./images/Screenshot_from_2021-10-22_13-59-52.png)
description

![text description](./images/Screenshot_2021-12-17_19-08-06.png)
description

![text description](./images/image.png)
description

![text description](./images/Screenshot_2022-03-05_12-03-42.png)
description

![text description](./images/Screenshot_2022-03-05_12-45-59.png)
description

![text description](./images/Screenshot_2022-03-05_12-47-42.png)
description

![text description](./images/Screenshot_2022-03-05_12-05-23.png)
description




## methods
- image processing operations
    + openCV
- image recognition
- object tracking
- sythetic image dataset
- image homographical transformation
- deep/machine learning
- video/play segmentation
- Image Processing
- Object detection (YOLO v4)
- Object interaction tracking
- Object motion tracking
- Multi-algorithm implementation (Detection -> tracking -> identification)
- Machine learning (Extrapolation; training against current data)
- Human pose estimation
- grid positioning
- 3D human interaction


## results

(sshot> image recognition, ball and humans)


(sshot> pitch 3d modelling)


(sshot> camera automation, in dataset)


## achievements and limits
### achievements

### limits
spatio-temporal data stream correction
human detection may contain more than 1 human
    consistent ball detection


- Human agent must verify and validate data collection
- Human agent must input match meta data
- Human agent must supervise/calibrate video processing
- Tracking broadcast is affected by zoom/replays and camera changes


## dissertation organisation sketch




# Background
        SKIP!!!!!!!!!!!!!!!!!!!!!




# Work carried out

- video processing
    + machine learning humans and ball recognition
    + object traking
- 3d modelling and dataset generation
    + pitch construction in blender
    + camera positioning
    + data generation scripting
    + homographical transformation



# Testing
## Testing assessment

testing with random camera, get accuracy

## Performance assessment

## any other experimental work




# Conclusions
## main achievements
        (relating them to initial objectives)
        (as well as similar worh from others)


## the main limitations of work
- detect ball consistently
- the output will always be an approximation
- video segment detection (also replays)
- calculate ball trajectory
- cannot detect players outside camera frame
- is not real-time, at this moment
- recognise video segments

## possible extensions and future work
- create data format (possibly logical ontology to leverage a logical reasoner) to create a richer dataset

- use Google Research Football Environment
    + predict off-screen player positioning
    + predict player actions
- action recognition
- modularise all modules and algorithms to allow other sports
- allow python running
