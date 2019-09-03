# Hand Movement Trajectories Tracking (Based on Colour Segmentation)
## Methodology
1. Convert video frame from RBG/BGR to HSV 
2. Apply skin detection by lower & upper thresh of skin color filtering
3. Apply morphology operations to et rid of the noisy specks.
4. Apply KNN background subtraction to refine skin filtering result, i.e. to further remove static skin coulor related background (face will be fading out, if it does not move)/ or a rectangular box is drawn around the face previously detected
5. Find contours of both hands by
   - Firstly sorting contours by area  (get the largest two contours i.e. get two hands out)
   - Secondly sorting contours by position (get hands from left to right )
   - Draw Convex Hull contour and normal contour
6. Tracking hand movement trajectories based on contour mass centroid 
7. Face detection is also performed using HAAR CASCADE Classifiers

<p align="center">
<img src="Image/Figure%201.PNG" width="650">
</p>

## Results
<div align="center">
  <a href="https://www.youtube.com/watch?v=nwIRszst49Y&feature=youtu.be&t=2"><img src="https://img.youtube.com/vi/nwIRszst49Y/hqdefault.jpg" alt="IMAGE ALT TEXT"></a>
</div>
<p align="center">
<img src="Image/Figure5left.PNG" width="250"><img src="Image/Figure5right.PNG" width="250">
</p>  
<p align="center">
<img src="Image/Figure6left.PNG" width="250"><img src="Image/Figure6right.PNG" width="250">
</p>

<p align="center">
<img src="Image/Figure6left2D.png" width="250"><img src="Image/Figure6right2D.png" width="250">
</p>  

## Version Notes
**v1.py** is hand movement trajectories tracking based on countour mass centriods with X vs Y trajectorires plot

**v2.py** is hand movement trajectories tracking based on countour mass centriods with X, Y, vs time 2D trajectorires plot

**v3.py** is hand movement trajectories tracking based on countour mass centriods with X, Y, vs time 3D trajectorires plot

**v4.py** multiple colour space filtering models with multi-colour thresholds (HSV/YCrCb/Lab/XYZ) for skin segmentation are considered. Also a rectangular box is drawn around the face (previously detected).

For the reference, the code in this repository has been tested on a desktop PC with:
* Python 3.6.5
* OpenCV 3.3.1
## Citations
```
@inproceedings{liang2019handtracking,
  author = {X. Liang, E. Kapetanios, B. Woll and A. Angelopoulou},
  booktitle = {Cross Domain Conference for Machine
Learning and Knowledge Extraction (CD-MAKE2019)},
  title = {Real Time Hand Movement Trajectory Tracking for Enhancing
Dementia Screening in Ageing Deaf Signers of British Sign Language},
  year = {2019}
}
```
