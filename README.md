# Hand Movement Trajectories Tracking
1. Convert video frame from RBG/BGR to HSV 
2. Apply skin detection by lower & upper thresh of skin color filtering
3. Apply morphology effects to denoise
4. Apply KKN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)
5. Find contours of both hands by
   - Firstly sorting contours by area  (get the largest two contours i.e. get two hands out)
   - Secondly sorting contours by position (get hands from left to right )
   - Draw Convex Hull contour and normal contour
6. Tracking hand movement trajectories based on contour mass centroid 
7. Face detection is also performed using HAAR CASCADE Classifiers



v1: is hand movement trajectories tracking based on countour mass centriods with X-Y trajectorires plot
