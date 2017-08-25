**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/car.png
[noncar]: ./output_images/noncar.png
[car_hog]: ./output_images/car_hog.png
[noncar_hog]: ./output_images/noncar_hog.png
[pipelineoutput]: ./output_images/pipelineoutput.png
[carboxes]: ./output_images/carboxes.png
[heatmap]: ./output_images/heatmap.png
[labels]: ./output_images/labels.png
[carboxesfinal]: ./output_images/carboxesfinal.png
[v1]: ./output_images/v1.png
[v2]: ./output_images/v2.png
[v3]: ./output_images/v3.png
[v4]: ./output_images/v4.png
[v5]: ./output_images/v5.png
[v6]: ./output_images/v6.png
[video1]: ./project_video.mp4


---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used the same method as in class, using the get_hog_features utils.py method. I tried several hog hyperparameters, and experimented with this method in one of the ipynb cells by visualizing several such HOG images both for cars and non cars. I found that increasing both the orient and pix_per_cell helps with detection, mostly eliminated many false positives I got without it. 


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car]
![alt text][noncar]
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Grayscale` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][car_hog]
![alt text][noncar_hog]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, and for the decision basically I looked how the visualization looks, then how is the model training score is affecte and ofcourse the final result.

I wanted to add both the spatial and color histo for the feature vector (enabling them to 'true' in the hyperparameter), but I found they introduce more false positive than being helpful. I wondered if I had same color-scaling bug, but didn't one.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the following hyper parameter for feature extraction:  
  color_space = 'YCrCb'
  orient = 13  # HOG orientations  
  pix_per_cell = 16 # HOG pixels per cell  
  cell_per_block = 2 # HOG cells per block  
  hog_channel = "ALL" 
  spatial_feat = False # Spatial features on or off  
  hist_feat = False # Histogram features on or off  
  hog_feat = True # HOG features on or off  
  test_set_size = 0.2  

This code can be found in the ipynb in cell number 5 and in 'extract_features' method in utils.py

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search can be found in cell #6 of the ipynb, in the method find_cars the sliding window mechanism can be found as seen in class. I chose larger scale for nearby to camera regions of the image, and smaller scales for 'farther' away down the image, as objects look smaller when they are far in perspective view.
The exact regions and scales chosen was an empiric process on the testing data, besides the insight mentioned above regarding scales.
Here is an example of detecting multiple boxes while sliding search, getting a heatmap, thresholding and getting labels, and final output.
![alt text][carboxes]
![alt text][heatmap]
![alt text][labels]
![alt text][carboxesfinal]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the YCrCB 3-channel HOG features, I discarded the spatiall binned and color histogram features as explained before. It provided for the best result for me whilst trying different options. Used multiple scales. Used the method that calculate HOG once per window as shown in class to help with performance.

![alt text][pipelineoutput]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps, labels and output:

![alt text][v1]
![alt text][v2]
![alt text][v3]
![alt text][v4]
![alt text][v5]
![alt text][v6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My main difficulty was by getting worse result while using the extra non HOG features, which I'm still not completely sure why.
In theory, using the color_his features should help even better to discard false positives, as some of the current false positives that randomly have the same HOG features, will probably will not also have the same color_his features.

The algorithm can be improved by:
* The data should be augmented with much more lightning conditions, and variations to make it robust to less nicer videos as I trained it on
* It is not aware of the width of the road, and sometimes detecting false positives in parallel road or outside the road.
* Adding more features than HOG
* Performace for videos isn't great and should be improved if it wants to be real-time software.
