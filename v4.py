import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Initialize webcam input
cap = cv2.VideoCapture(0)

# Initialize video input

####BSL Corpus#####
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/BSL Corpus Data Results/5/BL9i.MOV")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/BSL Corpus Data Results/4/BM17i.MOV")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/BSL Corpus Data Results/3/BF3i.MOV")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/BSL Corpus Data Results/2/G11c.MOV")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/BSL Corpus Data Results/1/BF14c.MOV")

####Sign Bank #####
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/SignBank Data Results/5/SO-WHAT.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/SignBank Data Results/4/POLICE.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/SignBank Data Results/3/BISCUIT.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/SignBank Data Results/2/AFRICA.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Desktop/Trajectoreis_test/SignBank Data Results/1/ACCEPT.mp4")

###################
cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Stomach/FARM.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Below Waist/LAP.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Restricted/L2n.mov")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Single Sign/Pronated Wrist/WATCH2.mp4")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Conversation/L12n.mov")
#cap = cv2.VideoCapture("C:/Users/liangx/Documents/Dunhill Project Data/Conversation/CF27l.mov")


# Set different colour conversion models {1 : HSV,2 : YCrCb,3 : LAB, 4 : XYZ,}
COLOUR_MODEL = 1;            
           
# Set Tracking Delay (i.e. delay in number of frames) to wait for KNN background subtraction work (Camera: 30; Video: 5)
DELAY= 10

# Set countour radius to denoise, only contours bigger enough are tracked (Camera: 45-55 ajust the value depending on distance between tracking object and camera; Video: 35)
RADIUS = 35

# Set frame count number for tracking trails reset (when there is no hands being detected)
FRAME = 100


# Initialize frame_acount
frame_count = 0


# Create empty points array for hand trajectories tracking 
points_left = []
points_right = []


# returns the elapsed milliseconds since the start of the program
def milliseconds():
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms

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

#Plot trajectories X-Y
def plot_trajectories(center,str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    plt.plot(xs, ys, color= clr)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(str + ' hand trajectories')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (opencv choose the coordinate system of points/images from Top-Left corner)
    #plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories like a Mirror View)
    plt.show()
    return None

#Plot trajectories with time
def plot_trajectories_vstime(center,str):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]
    plt.plot(ts, xs, color='b', marker ='o',label='$X-Trajectory$')
    plt.plot(ts, ys, color='y', marker ='^',label='$Y-Trajectory$')
    plt.xlabel('Time')
    plt.ylabel('X-Y')
    plt.title(str + ' hand trajectories')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (y reverted for:opencv choose the coordinate system of points/images from Top-Left corner; x reverted for: mirror effect)
    #plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories like a Mirror View)
    plt.legend(loc='upper right')
    plt.show()
    return None


#Plot 3D trjectories with Timeline in Z
def plot_trajectories_3d(center, str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]  
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    
    ax.plot3D(xs, ts, ys, color= clr, marker ='o') 
    #ax.set_yticks =(0, -1, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Time (ms)')
    ax.set_zlabel('Y')
    ax.set_title(str + '-Trajectory')
    plt.gca().invert_zaxis()  #Reverse Z-Axis in PyPlot (to revert y)
    #plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories like a Mirror View)
    plt.show()
    return None
 
def plot_trajectory_diagrams():
    plot_trajectories(points_left, "Left", "red")
    plot_trajectories(points_right, "Right", "green")
    plot_trajectories_vstime(points_left,(DATE+" Left"))
    plot_trajectories_vstime(points_right, (DATE+" Right"))  
    plot_trajectories_3d(points_left,(DATE+" Left"),  "red")
    plot_trajectories_3d(points_right,(DATE+" Right"), "green")
    return None
   

# define the different colour conversion function blocks: from RBG/BGR to HSV/YCrCb/LAB/XYZ 
def HSV():
    con_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV Colour Model Image:', con_img)
    return con_img

def YCrCb():
    con_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    cv2.imshow('YCrCb Colour Model Image:', con_img)
    return con_img

def LAB():
    con_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    cv2.imshow('CIE LAB Colour Model Image:', con_img)
    return con_img

def XYZ():
   con_img = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
   cv2.imshow('CIE XYZ Colour Model Image:', con_img)
   return con_img

# map the inputs to the different colour convertion function blocks
options = {1 : HSV,
           2 : YCrCb,
           3 : LAB,
           4 : XYZ,
}

# define the different colour convertion threshold
def HSV_thre():
    # Selected Value sets
    low_thresh = np.array([0, 48, 80], dtype = "uint8")
    up_thresh = np.array([20, 255, 255], dtype = "uint8")
    
    return low_thresh, up_thresh

def YCrCb_thre():
    # Selected Value sets
    low_thresh = np.array((0, 133,77), dtype = "uint8")
    up_thresh = np.array((255, 173,127), dtype = "uint8")

    return low_thresh, up_thresh

def LAB_thre():
    # Selected Value sets
    low_thresh = np.array((20, 128, 130), dtype = "uint8")
    up_thresh = np.array((220, 245, 255), dtype = "uint8")

    return low_thresh, up_thresh

def XYZ_thre():

    low_thresh = np.array((79, 80, 30), dtype = "uint8")
    up_thresh = np.array((240, 240,140), dtype = "uint8")
    
    return low_thresh, up_thresh

# map the inputs to different colour convertion threshold
options_thre = {1 : HSV_thre,
                2 : YCrCb_thre,
                3 : LAB_thre,
                4 : XYZ_thre,
}

#Set lower_thresh, upper_thresh for different colour convertion models
lower_thresh, upper_thresh = options_thre[COLOUR_MODEL]()

# Get current date & time
DATE= datetime.now().strftime('%Y:%m:%d')
start_time = datetime.now()

# Loop video capture until break statement is exectured
while cap.isOpened(): 
    
    # Read webcam/video image
    ret, frame = cap.read()
    
    # when there is a video input
    if ret == True:

        # Get default camera/video window size
        Height, Width = frame.shape[:2]
       
        #Different colour convertion function blocks is invoked:
        converted_img = options[COLOUR_MODEL]()
       
        
        # Face Detection Using HAAR CASCADE 
        hc_face = cv2.CascadeClassifier("C:/Users/liangx/source/repos/Skin Detection/haarcascade_frontalface_alt/haarcascade_frontalface_alt.xml")
        faces = hc_face.detectMultiScale(converted_img)
        for (x,y,w,h) in faces:
            # If we do not draw a box on face, then use the code below 
            #cv2.rectangle(converted_img, (x,y), (x+w,y+h), 255, thickness=2)

            # If we draw a box on face to avoid face skin detection, then use the code below 
            cv2.rectangle(converted_img, (x-10,y-30), (x+w+10, y+h+80), (255,255,255), -1)
            crop_img = frame[y+2:y+w, x+2:x+h]
            cv2.imshow('Face Detection', crop_img)
            
       
        # Use inRange to capture only the values between lower & upper_thresh for skin detection
        mask = cv2.inRange(converted_img, lower_thresh, upper_thresh) 
        
        # Adding morphology effects to denoise
        kernel_morphology =np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel_morphology, iterations=1)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_morphology)
        mask = cv2.dilate(mask, kernel_morphology, iterations=1)
        cv2.imshow('Skin colour + Morpho Mask', mask)

        # Perform Bitwise AND on mask and original frame
        # rest1 is the results after applying morphology effects + skin filtering
        rest1 = cv2.bitwise_and(frame, frame, mask= mask)
        

        # Find contours on mask
        # cv2.RETR_EXTERNAL finds external contours only; cv2.CHAIN_APPROX_SIMPLE only provides start and end points of bounding contours, thus resulting in much more efficent storage of contour information.
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print ("Number of contours1 found = ", len(contours))
   

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
                cv2.drawContours(rest1, [hull], -1, (0,0,255), 3)
                      
                # Draw Normal Contour
                cv2.drawContours(rest1, [c], -1, (255,0,0), 3)

                         
                # Show hands Contour
                cv2.imshow('Contours by area', rest1)


                # Tracking Left hand   
                if i == 0:
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    
                    #3D Plot in (mili second) Format
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), milliseconds())
                    
                    # Draw cirlce and leave the last center creating a trail
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 0, 255), 2)

                    
                    # Only contours with radius > RADIUS are tracked (de-noise)
                    if  radius > RADIUS:
                        
                        points_left.append(center)
                        # loop over the set of tracked points to draw tracking lines (starts with frames delay- to wait for KNN background subtraction work)
                        for l in range(DELAY, len(points_left)):
                            try:
                                cv2.line(frame, points_left[l - 1][:2], points_left[l][:2], (0, 0, 255), 2)
                            except:
                                pass
                        frame_count = 0
                    else:
                         frame_count += 1
                         # If there is no hand detected,  when count frames to FRAME, plot trajectories before clear the trajectories trails
                         if frame_count == FRAME:
                            #print("frame_count",frame_count)                   
                            plot_trajectory_diagrams()
                            points_left = []
                            points_right = [] 
                            frame_count = 0
                        
                            
                # Tracking Right hand  
                else:    
                    (x, y), radius = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]),  milliseconds())
                    # Draw cirlce and leave the last center creating a trail
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 0), 2)
                   
                    # loop over the set of tracked points
                    if  radius > RADIUS:
                        points_right.append(center)
                        for l in range(DELAY, len(points_right)):
                            try:
                                cv2.line(frame, points_right[l - 1][:2], points_right[l][:2], (0, 255, 0), 2)
                            except:
                                pass
                        frame_count = 0   

                    else:
                        frame_count += 1
                        # If there is no hand detected,  when count frames to FRAME, plot trajectories before clear the trajectories trails
                        if frame_count == FRAME:
                            #print("frame_count",frame_count)
                            plot_trajectory_diagrams()
                            points_left = []
                            points_right = [] 
                            frame_count = 0                      

        else:
          pass


        # Display our object tracker
        #frame = cv2.flip(frame, 1)
        cv2.imshow("Object Tracker", frame)
       


        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectory_diagrams()
            break

    else:
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectory_diagrams()
            break

cap.release()
cv2.destroyAllWindows() 
