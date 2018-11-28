# HandTracking
1. Convert video frame from RBG/BGR to HSV 
2. Skin detection based on lower & upper_thresh of skin HSV range
3. Apply morphology effects to denoise
4. Apply KKN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)
5. Find contours of both hands by
 - First sorting contours by area  (Get the largest two contours i.e. two hands)
 - Then sorting contours by position (get left to right hands)
6.  Tracking hand movement trajectories by contour mass centroid 
7. Also face is detected using HAAR CASCADE
