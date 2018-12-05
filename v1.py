#Xing @ 2018.12.05
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
 

# Initialize webcam input
cap = cv2.VideoCapture(1)

# Initialize video input
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Pronated Wrist/WATCH2.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Stomach/FARM.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Below Waist/LAP.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Restricted/L2n.mov")

# Set Tracking Delay (i.e. delay in number of frames) to wait for KNN background subtraction work (Camera: 30; Video: 5)
DELAY= 5

# Set countour radius to denoise, only contours bigger enough are tracked (Camera: 45-55 ajust this value depending on distance between tracking object and camera; Video: 35)
RADIUS = 55

# Set frame count number for tracking trails reset (when there is no hands being detected)
FRAME = 100

# Initialize frame_acount
frame_count = 0

# define range of skin color in HSV (works good with brown skin)
lower_thresh = np.array([0, 48, 80], dtype = "uint8")
upper_thresh = np.array([20, 255, 255], dtype = "uint8")

# Create empty points array for hand trajectories tracking 
points_left = []
points_right = []

# Initlaize K-Nearest Neighbors (KNN) background subtractor
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

#Plot trajectores
def plot_trajectories(center,str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    plt.plot(xs, ys, color= clr)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(str + ' hand trajectores')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (OpenCv choose the coordinate system of points/images from Top-Left corner)
    plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories look like a Mirror View)
    plt.show()
    return None

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
        cv2.imshow('Skin colour + Morpho Mask', mask)

        # Perform Bitwise AND on mask and original frame
        # rest1 is the results after applying morphology effects + skin filtering
        rest1 = cv2.bitwise_and(frame, frame, mask= mask)
        
        # Apply KKN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move) 
        fgmask = fgbg.apply(rest1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_bgsub)
        cv2.imshow('Background subtraction + Skin colour + Morpho Mask',fgmask) 

        # Perform Bitwise AND on fgmask and rest1 frame
        # rest2 is results after applying background subtraction + morphology effects  + skin filtering
        rest2 = cv2.bitwise_and(rest1, rest1, mask= fgmask)

        # Find contours 
        # cv2.RETR_EXTERNAL finds external contours only; cv2.CHAIN_APPROX_SIMPLE only provides start and end points of bounding contours, thus resulting in much more efficent storage of contour information.
        _, contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print ("Number of contours1 found = ", len(contours))
        #print(type(contours))   #The variable 'contours' are stored as a numpy array of (x,y) points that form the contour
        
        
        # Draw all Contours found
        #cv2.drawContours(rest2, contours, -1, (0,255,0), 3)
        #cv2.imshow('All Contours filtered by skin color and background subtraction', rest2)  
        #cv2.imshow('Original', frame)  
   

        # When both hands are detected 
        if len(contours) >=2:
        
            # Get the largest two contours and its center (i.e. two hands)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            # Sort by reverse=True, using our x_cord_contour function (i.e. hands tracking from left to right)
            contours_left_to_right = sorted(sorted_contours, key = x_cord_contour, reverse = True)

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
                        # loop over the set of tracked points to draw tracking lines (starts with frames delay- to wait for KNN background subtraction work)
                        for l in range(DELAY, len(points_left)):
                            try:
                                cv2.line(frame, points_left[l - 1], points_left[l], (0, 0, 255), 2)
                            except:
                                pass
                        frame_count = 0
                    else:
                         frame_count += 1
                         # If there is no hand detected,  when count frames to FRAME, plot trajectories before clear the trajectories trails
                         if frame_count == FRAME:
                            print("frame_count",frame_count)                   
                            plot_trajectories(points_left,"Left", "red")
                            plot_trajectories(points_right, "Right", "green")                           
                            points_left = []
                            points_right = [] 
                            frame_count = 0
                        
                            
                # Tracking Right hand  
                else:    
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
                        frame_count += 1
                        # If there is no hand detected,  when count frames to FRAME, plot trajectories before clear the trajectories trails
                        if frame_count == FRAME:
                            print("frame_count",frame_count)
                            plot_trajectories(points_left, "Left", "red")
                            plot_trajectories(points_right, "Right", "green")
                            points_left = []
                            points_right = [] 
                            frame_count = 0                      

        else:
          pass


        # Display our object tracker
        frame = cv2.flip(frame, 1)
        cv2.imshow("Object Tracker", frame)
       


        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectories(points_left, "Left", "red")
            plot_trajectories(points_right, "Right", "green")
            break

    else:
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectories(points_left, "Left", "red")
            plot_trajectories(points_right, "Right", "green")           
            break

cap.release()
cv2.destroyAllWindows() 
 
