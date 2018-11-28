# Xing @ 2018.11.27
import cv2
import numpy as np
import time

# Initialize webcam input
cap = cv2.VideoCapture(1)

# Initialize video input
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Pronated Wrist/WATCH2.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Stomach/FARM.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Below Waist/LAP.mp4")

# Set Tracking Delay (i.e. delay in number of frames) to wait for KNN background subtraction work (Camera: 25; Video: 5)
DELAY= 5

# Set countour radius to denoise, only contours which is big enough will be tracked 
RADIUS = 25

# Set frame count for tracking trails reset, if there is no hands being detected
frame_count = 0

# Get video/camera input details
"""
lengthVideo = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
widthVideo  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
heightVideo = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fps for video input with a stream header (need nb_frames field) 
fpsVideo = int(cap.get(cv2.CAP_PROP_FPS)) 

# fps for camera input without a stream header 
# Number of frames to capture
num_frames = 120;         
print ("Capturing {0} frames".format(num_frames))
# Start time
start = time.time()
# Grab a few frames
for i in range(0, num_frames):
    ret, frame = cap.read()
# End time
end = time.time()
# Time elapsed
seconds = end - start
print ("Time taken : {0} seconds".format(seconds))
# Calculate frames per second
fpsCamera  = int(num_frames / seconds)
print ("Estimated frames per second : {0}".format(fpsCamera))    
"""

# define range of skin color in HSV
#lower_thresh = np.array([0, 50, 0])
#upper_thresh = np.array([120, 150, 255])

# define range of skin color in HSV (works better with brown skin)
lower_thresh = np.array([0, 48, 80], dtype = "uint8")
upper_thresh = np.array([20, 255, 255], dtype = "uint8")

# Create empty points array for hand trajectories tracking 
points_left = []
points_right = []

# Initlaize KNN background subtractor
kernel_bgsub = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorKNN()

# Sorting contour by area
def get_contour_areas(contours):  
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas

# Sorting contour by position
def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    M = cv2.moments(contours)
    return (int(M['m10']/M['m00']))
    

# CAMshift first frame window initialize 
"""
# take first frame of the video
ret, frame = cap.read()

# setup default location of window
r, h, c, w = 240, 100, 400, 160 
track_window = (c, r, w, h)

# Crop region of interest for tracking
roi = frame[r:r+h, c:c+w]

# Convert cropped window to HSV color space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, lower_thresh, upper_thresh)

# Obtain the color histogram of the ROI
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

# Normalize values to lie between the range 0, 255
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria
# We stop calculating the centroid shift after ten iterations 
# or if the centroid has moved at least 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
"""


# Loop video capture until break statement is exectured
while cap.isOpened(): 
    
    # Read webcam/video image
    ret, frame = cap.read()
    
    # when there is a video input
    if ret == True:

        # Get default camera/video window size
        Height, Width = frame.shape[:2]
        
       
        # Convert image from RBG/BGR to HSV 
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Face Detection Using HAAR CASCADE 
        hc_face = cv2.CascadeClassifier("C:/Users/liangx/source/repos/Skin Detection/haarcascade_frontalface_alt/haarcascade_frontalface_alt.xml")
        faces = hc_face.detectMultiScale(hsv_img)
        for (x,y,w,h) in faces:
            cv2.rectangle(hsv_img, (x,y), (x+w,y+h), 255, thickness=2)
            crop_img = frame[y+2:y+w, x+2:x+h]
            cv2.imshow('Face Detection', crop_img)
 
        
        # Use inRange to capture only the values between lower & upper_thresh for skin detection
        mask = cv2.inRange(hsv_img, lower_thresh, upper_thresh) 
        
        # Adding morphology effects to denoise
        kernel_morphology =np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel_morphology, iterations=1)
        #mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel_morphology)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_morphology)
        mask = cv2.dilate(mask, kernel_morphology, iterations=1)
        cv2.imshow('Skin colour Mask', mask)

        # Perform Bitwise AND on mask and original frame
        # rest1 is the results after applying morphology effects
        rest1 = cv2.bitwise_and(frame, frame, mask= mask)
        
        # Apply KKN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move) 
        fgmask = fgbg.apply(rest1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_bgsub)
        cv2.imshow('Background subtraction Mask',fgmask) 

        # Perform Bitwise AND on fgmask and rest1 frame
        # rest2 is results after applying background subtraction and morphology effects
        rest2 = cv2.bitwise_and(rest1, rest1, mask= fgmask)

        # Find contours 
        # cv2.RETR_EXTERNAL finds external contours only; cv2.CHAIN_APPROX_SIMPLE only provides start and end points of bounding contours, thus resulting in much more efficent storage of contour information.
        _, contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print ("Number of contours1 found = ", len(contours))
        #print(type(contours))   #The variable 'contours' are stored as a numpy array of (x,y) points that form the contour
        
        
        # Find Canny Edges (not helpful!)
        """
        edged = cv2.Canny(rest2, 30, 240)
        cv2.imshow('Canny Edges', edged)
        
        # Find contours 
        # cv2.RETR_EXTERNAL finds external contours only
        # cv2.CHAIN_APPROX_SIMPLE only provides start and end points of bounding contours, thus resulting in much more efficent storage of contour information.
        _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print ("Number of contours2 found = ", len(contours))
        #print(type(contours))   #The variable 'contours' are stored as a numpy array of (x,y) points that form the contour 
        #Draw all contours
        """
        
        # Draw all Contours found
        #cv2.drawContours(rest2, contours, -1, (0,255,0), 3)
        #cv2.imshow('All Contours filtered by skin color and background subtraction', rest2)
        
        #cv2.imshow('Original', frame)  


        # CAMshift
        """
        hsv= cv2.cvtColor(rest2, cv2.COLOR_BGR2HSV) 
        # Calculate the histogram back projection 
        # Each pixel's value is it's probability
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply Camshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image 
        # We use polylines to represent Adaptive box 
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        
        cv2.imshow('Camshift Tracking', img2)
        """

        # Create empty centre array to store centroid center of mass
        center = int(Height*2/3), int(Width*2/3)

        # When both hands are detected 
        if len(contours) >=2:
        
            # Get the largest two contours and its center (i.e. two hands)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            # Sort by left to right using our x_cord_contour function (i.e. hands tracking from left to right)
            contours_left_to_right = sorted(sorted_contours, key = x_cord_contour, reverse = False)

            # Iterate over two contours and draw one at a time
            for (i,c) in enumerate(contours_left_to_right):
                
                # Draw Convex Hull Contour   
                hull=cv2.convexHull(c)
                cv2.drawContours(rest2, [hull], -1, (0,0,255), 3)
                      
                # Draw Normal Contour
                cv2.drawContours(rest2, [c], -1, (255,0,0), 3)
                
                # Show hands Contour
                cv2.imshow('Contours by area', rest2)


                # Tracking Left hand   
                if i == 0:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # Draw cirlce and leave the last center creating a trail
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)
                    
                    # Only contours with radius > RADIUS are tracked (de-noise)
                    if  radius > RADIUS:
                        points_left.append(center)
                        # loop over the set of tracked points to draw tracking lines (starts with frames delay- approximately 1 seconds delay to make KNN background subtraction work)
                        for l in range(DELAY, len(points_left)):
                            try:
                                cv2.line(frame, points_left[l - 1], points_left[l], (0, 0, 255), 2)
                            except:
                                pass
                        frame_count = 0
                    else:
                         # If there is no hand detected, count frames 
                        frame_count += 1
                        print("frame_count",frame_count)
                        # when frame_count reaches 20, clear our trails
                        if frame_count == 5:
                            points_left = []
                            points_right = [] 
                            frame_count = 0
       
                            
                # Tracking Right hand  
                elif i == 1:    
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    # Draw cirlce and leave the last center creating a trail
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 0), 2)
                   
                    # loop over the set of tracked points
                    if  radius > RADIUS:
                        points_right.append(center)
                        for l in range(DELAY, len(points_right)):
                            try:
                                cv2.line(frame, points_right[l - 1], points_right[l], (0, 255, 0), 2)
                            except:
                                pass
                        frame_count = 0   

                    else:
                        # If there is no hand detected, count frames 
                        frame_count += 1
                        # when frame_count reaches 50, clear our trails
                        if frame_count == 50:
                            points_left = []
                            points_right = [] 
                            frame_count = 0


                else:
                     pass


        else:
          pass


        # Display our object tracker
        frame = cv2.flip(frame, 1)
        cv2.imshow("Object Tracker", frame)

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    else:
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

cap.release()
cv2.destroyAllWindows() 
 
