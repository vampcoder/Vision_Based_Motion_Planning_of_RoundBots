# Generating-2D-Map-From-Multiple-Camera-Using-Homography
In computer vision, we define planar homography as a projective mapping from one plane to
another. Thus, the mapping of points on a two-dimensional planar surface to the image of our 
camera is an example of planar homography.

Here we are trying to generate 2-D map using multiple camera. Here we are taking input from two cameras and converting them
Overhead images, and combining them into single bigger image.

Here we have three codes, first CalibMatrices is used to get required transformation matrices, which is stored
next we have LiveRunningOverhead.py which generates overhead images for the images taken from cameras in that arrangement.
and LiveRunningOverheadVideo.py generates similar map but it runs live by taking video input.

After the map is generated it can be used as a input map and planning algorithms like A*, Artificial potential field, etc can be applied.

---

<b> Prerequisites</b>
- PIL
- Python
- OpenCV
- Numpy

<b> Input Images </b>
It will take all images in input folder as input images.

Input Images camera1:

![alt text][logo1]
[logo1]: input/camera1/picture064.jpg "Sample Image"

![alt text][logo2]
[logo2]: input/camera1/picture063.jpg "Sample Image"

![alt text][logo3]
[logo3]: input/camera1/picture060.jpg "Sample Image"

![alt text][logo4]
[logo4]: input/camera1/picture059.jpg "Sample Image"

Input Images camera2:

![alt text][logo5]
[logo5]: input/camera2/picture065.jpg "Sample Image"

![alt text][logo6]
[logo6]: input/camera2/picture062.jpg "Sample Image"

![alt text][logo7]
[logo7]: input/camera2/picture061.jpg "Sample Image"

![alt text][logo8]
[logo8]: input/camera2/picture058.jpg "Sample Image"

Output Images:

![alt text][logo9]
[logo9]: output/output3.jpg "Sample Image"

![alt text][logo10]
[logo10]: output/FinalImage.jpg "Sample Image"

![alt text][logo11]
[logo11]: output/output31.jpg "Sample Image"

![alt text][logo12]
[logo12]: output/FinalImage2.jpg "Sample Image"

![alt text][logo13]
[logo13]: output/output32.jpg "Sample Image"

![alt text][logo15]
[logo15]: output/FinalImage3.jpg "Sample Image"

![alt text][logo16]
[logo16]: output/output33.jpg "Sample Image"

![alt text][logo17]
[logo17]: output/FinalImage4.jpg "Sample Image"

