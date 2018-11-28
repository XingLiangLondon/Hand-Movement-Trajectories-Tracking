# Hand Movement Trajectories Tracking
1. Convert video frame from RBG/BGR to HSV 
2. Apply skin detection based on lower & upper thresh of skin color filtering
3. Apply morphology effects to denoise
4. Apply KKN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)
5. Find contours of both hands by
   - Firstly sorting contours by area  (Get the largest two contours i.e. two hands)
   - Secondly sorting contours by position (get hands from left to right )
6. Tracking hand movement trajectories based on contour mass centroid 
7. Face detection is also performed using HAAR CASCADE
