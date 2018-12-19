# Hand Movement Trajectories Tracking
1. Convert video frame from RBG/BGR to HSV 
2. Apply skin detection by lower & upper thresh of skin color filtering
3. Apply morphology effects to denoise
4. Apply KNN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)
5. Find contours of both hands by
   - Firstly sorting contours by area  (get the largest two contours i.e. get two hands out)
   - Secondly sorting contours by position (get hands from left to right )
   - Draw Convex Hull contour and normal contour
6. Tracking hand movement trajectories based on contour mass centroid 
7. Face detection is also performed using HAAR CASCADE Classifiers


Just for reference, the code in this repository has been tested on a desktop PC with:
- Python 3.6.5
- OpenCV 3.3.1

v1.py: is hand movement trajectories tracking based on countour mass centriods with X vs Y trajectorires plot

v2.py: is hand movement trajectories tracking based on countour mass centriods with X, Y, vs time trajectorires plot
