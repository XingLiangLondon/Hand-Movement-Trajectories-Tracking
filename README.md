# Hand Movement Trajectories Tracking
## Methdology
* Convert video frame from RBG/BGR to HSV 
* Apply skin detection by lower & upper thresh of skin color filtering
* Apply morphology operations to denoise
* Apply KNN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)/ or a rectangular box is drawn around the face previously detected
* Find contours of both hands by
   - Firstly sorting contours by area  (get the largest two contours i.e. get two hands out)
   - Secondly sorting contours by position (get hands from left to right )
   - Draw Convex Hull contour and normal contour
* Tracking hand movement trajectories based on contour mass centroid 
* Face detection is also performed using HAAR CASCADE Classifiers


Just for reference, the code in this repository has been tested on a desktop PC with:
* Python 3.6.5
* OpenCV 3.3.1
## Version Notes
v1.py: is hand movement trajectories tracking based on countour mass centriods with X vs Y trajectorires plot

v2.py: is hand movement trajectories tracking based on countour mass centriods with X, Y, vs time 2D trajectorires plot

v3.py: is hand movement trajectories tracking based on countour mass centriods with X, Y, vs time 3D trajectorires plot

v4.py: multiple colour space filtering models with multi-colour thresholds (HSV/YCrCb/Lab/XYZ) for skin segmentation are considered. Also a rectangular box is drawn around the face (previously detected).
## Citations
@inproceedings{liang2019handtracking,
  author = {X. Liang, E. Kapetanios, B. Woll and A. Angelopoulou},
  booktitle = {CD-MAKE},
  title = {Real Time Hand Movement Trajectory Tracking for Enhancing
Dementia Screening in Ageing Deaf Signers of British Sign Language},
  year = {2019}
}
