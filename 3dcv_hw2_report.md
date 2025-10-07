---
title: 3dcv hw2

---

# 3dcv hw2
## How to run:
please install following package:
```
python
pandas
numpy
scipy
open3d
matplotlib
opencv-python
tqdm
```

requirements.txt is also in repo:
```
pandas
numpy
scipy
open3d
matplotlib
opencv-python
tqdm
```

I installed these using anaconda:
```
conda create -n myenv python=3.12
conda activate myenv
pip install -r requirements.txt
```

run:
```
python hw2.py
```
to get results printed out in console and written at file results.txt, visulaiztion of open3D should popup a window. Also video for Q2-2 will be in ar_cube_video_pnp.mp4

run:
```
python hw2_bonus.py
```
to get P3P+RANSAC BONUS results printed out in console and written at file results_bonus.txt, visulaiztion of open3D should popup a window. Also video for Q2-2 will be in ar_cube_video_bonus.mp4

## Problem 1
### Methods
#### 1-1
For 1-1, first use ffmpeg to get frames from recorded video with 2 FPS.
![image](https://hackmd.io/_uploads/SyaDHvChel.png)

Then put those frame into colmap to reconstruct the scene with these options.
I also selected dense option because it will make mesh in 1-2 looks better.
![螢幕擷取畫面 2025-10-04 171629](https://hackmd.io/_uploads/ryaiLDR3eg.png)

After a while we will get this sparse reconstruction displayed.
![image](https://hackmd.io/_uploads/rkrCrDR3gx.png)

#### 1-2
After I got dense reconstruction from colmap, I imported it to meshlab to edit and create mesh.
Then select Filters > Remeshing, Simplification and Reconstruction > Surface Reconstruction: Screened Poisson
![螢幕擷取畫面 2025-10-04 172439](https://hackmd.io/_uploads/r1fmOPC2xl.png)
This will create a basic mesh for us.

Then I used Select vertices function to select redundent vertices created by colmap to make the mesh looks clean.
![image](https://hackmd.io/_uploads/Bkc9dv03gl.png)

After that, used Filters > Smoothing, Fairing and Deformation > Laplacian Smooth to soften noisy areas.
Then I got my final mesh.
![image](https://hackmd.io/_uploads/BJ4btP02ll.png)

### Youtube link
https://youtu.be/MVkaO4GgYG0

## Problem 2
### Methods
#### 2-1
For pnpsolver(), I first used cv2.FlannBasedMatcher to match the discriptors, then use ratio test (0.7) to get good matches.
After that, I used cv2.solvePnPRansac with all camera parameters to get Rotation and Translation in vector form.

Once those values are returned from pnpsolver(), I transfered R into quaternion form and calculate error for R and T.

For rotation_error(), first calculate relative rotation from ground truth to computed rotation.
Then tranfer the relative rotation to axis-angle by as_rotvec(). And since magnitude of this vector specifies the angle of rotation (in radians), I returned np.degrees of np.linalg.norm(rotvec) to get rotation error.

For translation_error(), simply use np.linalg.norm(t1 - t2) will do.

Then after I got all errors for all validation images, I calculated median and calculate Camera2World_Transform_Matrixs. 
I've also calculated mean as well for reference.
![image](https://hackmd.io/_uploads/ryBF2xb6gg.png)
![image](https://hackmd.io/_uploads/SJlV9nxWTge.png)
![螢幕擷取畫面 2025-10-06 151627](https://hackmd.io/_uploads/SyKZRg-6ge.png)


For Camera2World_Transform_Matrixs, since I have quaternion of rotation and translation for all camera angles of world to camera, I get rotation for world to camera (R_c2w) by inverse of R and -R_c2w @ t_w2c for t_c2w.
Then to put them into 4x4 Camera2World_Transform_Matrixs, put R_c2w at top left and put t_c2w to last column.

To create visualization by open3D, first load point clouds into it. 
Then for each camera location, create pyramid by defining base and apex as:
![image](https://hackmd.io/_uploads/Hkc1epl6ex.png)
Then apply transform using already competed Camera2World_Transform_Matrixs and get world coordinates for the pyramid and draw it as lineset.

For creating trajectory line, just use the last column's first 3 value as world coordinates for each Camera2World_Transform_Matrixs as points for each camera position. Then connect points from one to the next.

Finally, since the point cloud are reconstructed using colmap I suppose? The scene is upside down and a little bit tilt. So I transform it to make it look normal.


#### 2-1 BONUS
To implement P3P + RANSAC myself, I used the paper [p3p made easy](https://arxiv.org/abs/2508.01312) as reference.
To implement P3P. RANSAC just follow the slides.

##### P3P
The overall algorithm for P3P is as follows:
![image](https://hackmd.io/_uploads/r1ngpy-plx.png)

First accoeding to it, we need to create unit bearing vector for the image key points(m), so I did that by first make it 3D and make it unit length.
After that the paper states that for every 3 points we selected, we should re-arrnage it so m13<=m12<=m23 where mij = dot(mi,mj)
![image](https://hackmd.io/_uploads/rklejygbaee.png)

Then we could compute s12,s13,s23,c4,c3,c2,c1,c0,A,B,C as written in the paper.
![image](https://hackmd.io/_uploads/BygAYJxZTle.png)

After that we need to find real root for quartic of (c4,c3,c2,c1,c0). 
Based on abs(c3 / c4), if it's bigger than 10 then we will use Ferrari-Lagrange method to solve the quartic, else we use Classical Ferrari method.
If they failed to return results, we will fallback to use np.roots() to find numeric roots.

---
**Classical Ferrari and Ferrari-Lagrange**

For classical Ferrari, we have to compress the original quartic into different form of:
![image](https://hackmd.io/_uploads/B1JplxWaee.png)
Then we can try to solve a resolvent cubic, which should guarentee to have a real root:
`u^3 + (5p/2) u^2 + (2r - 0.5p^2) u - (q^2)/8 = 0`
After we find real roots and get the biggest one(z), we can make the original quartic becomes:
```
(y^2 + sqrt(2z) y + (p/2 + z + q/(2*sqrt(2z)))*(y^2 - sqrt(2z) y + (p/2 + z - q/(2*sqrt(2z)))
```
Then use the formula we've learned in junior high to solve for y and finally substitude to get x, which is the real root for the quartic equation.


For Ferrari-Lagrange, the compression for the original quartic is the same, but the resolvent cubic we formed is:
```
u^3 + (2*p) u^2 + (p^2 - 4r) u - q^2 = 0
```
Which again we get the biggest real root from it and we can make the original quartic becomes:
```
(y^2 + alpha*y + ( (p/2) - (q/(2*alpha)) + (u/2) ))*(y^2 - alpha*y + ( (p/2) + (q/(2*alpha)) + (u/2) ) = 0)
where alpha = np.sqrt(max(0.0, u))
```
Again apply the formula we've learned in junior high to solve for y and finally substitude to get x, which is the real root for the quartic equation.

---

After we get real roots from either Classical Ferrarr or Ferrari-Lagrange, we compute d1,d2,d3 as stated in the paper:
![image](https://hackmd.io/_uploads/rkN5flWaeg.png)

Then we perform Gauss-Newton refinement on d1,d2,d3 using residual also written in the paper:
![image](https://hackmd.io/_uploads/rJK6MgWael.png)

Finally we can get Y1,Y2,Y3,X1,X2,X3 for computing rotation and translation for P3P by:
![image](https://hackmd.io/_uploads/B17fQxWplg.png)

##### RANSAC
For RANSAC, it's pretty straight forward, I first set the upper bound of iter and also the success rate of 0.99.
Then in each iter, just randomly sample 3 points and perform P3P written above to get the R,T.
Use such R,T to project 3D points onto 2D image and then calculate error between corresponding points in Euclidean Distance to decide if that point can be count as inliners.

Based on the size of inliners, it decide whether to early terminate the loop by:
![image](https://hackmd.io/_uploads/ry9e4lWaxl.png)

When the loop breaks or until 1000 loops then we can get our R,T for the camera location!

The rest of the code besides P3P and RANSAC is the same as 2-1.
![image](https://hackmd.io/_uploads/rylwlZZTxg.png)
![image](https://hackmd.io/_uploads/r10wxZ-plg.png)
![image](https://hackmd.io/_uploads/HJTNx--Tgg.png)


#### 2-2
To create VR video, first we need to sort the images by their real sequence since the original data sort them by their name.
This will leads to valid_img100 being before valid_img80, which is incorrect.
So I sorted them using:
![image](https://hackmd.io/_uploads/ryiYO0xpge.png)

Then I need to create a cube, so I write a function create_cube_voxels() which simply create a 10\*10\*10 cube and gives them color.
After that used cube_transform_mat.npy from transform_cube.py to transform those cube points into world coordinates.

Then for each valid image, I have the computed rotation and transform from Q2-1. So using it I can again transform cube points from world coordinates to camera coordinates.
After that just simply sort by their depth and use cv2.projectPoints() to project those 3d camera coordinate points onto the image and draw them.
I've also write a function to check that if the projected points should be draw on the image. Some of them will be projected into wierd shapes due to extreme position of points or camera angle. So having an extra safety is better.

Finally, I used cv2.VideoWriter to write images into a video.


### Youtube link
https://youtu.be/jMiddKHw3Gg

I've also uploaded the video output for Q2-2, since I notice when I was recording, my computer is pretty lagged, so it cause the video being played also lagged. But I played it after recording then it's pretty smooth:

https://youtube.com/shorts/5e5BSNW4B3I


#### Appendix
Thanks for 簡晟琪 for telling me about the paper P3P made easy.

I've used chatGpt-5 for helping me doing this homework.
I've used it to generate the overall structure of the code, then I will fix the bugs it created and also re-write some of it to make it matches the requirements of the homework.