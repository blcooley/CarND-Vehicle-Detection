**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./training/vehicles/GTI_Right/image0027.png
[image2]: ./training/non-vehicles/Extras/extra12.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/nice_result.png
[image5]: ./output_images/heat4.png
[image6]: ./output_images/heat5.png
[image7]: ./output_images/heat6.png
[image8]: ./output_images/heat7.png
[image9]: ./output_images/heat8.png
[image10]: ./output_images/heat9.png
[image11]: ./output_images/label4.png
[image12]: ./output_images/label5.png
[image13]: ./output_images/label6.png
[image14]: ./output_images/label7.png
[image15]: ./output_images/label8.png
[image16]: ./output_images/label9.png
[image17]: ./output_images/final_img4.png
[image18]: ./output_images/final_img5.png
[image19]: ./output_images/final_img6.png
[image20]: ./output_images/final_img7.png
[image21]: ./output_images/final_img8.png
[image22]: ./output_images/final_img9.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first two code cells of the IPython notebook. The first cell sets up the function `get_hog_features` to extract features from the images. The second code cell is a bit of a playground for visualizing the images.

In the second code cell, I used `glob.glob` to read in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1] ![alt text][image2]

I played around with various combinations of preprocessing (different colorspaces, orientation bins, pixels per cell, and cells per block). Truthfully, I was unable to get much of a feel for which colorspace and combinations of parameters would work well just from visualizing, so I implemented the support vector classifier and used its output to help me decide.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and evaluated their performance in the SVC (see next section) to determine which parameter sets to use. I evaluated the parameters according to the test error of the SVC and performance on test images provided in the test_images folder of the repository. I found that varying the orientation, pixels per cell, and cells per block from the defaults of 9, 8, and 2 (used in the Lesson) to be detrimental. Either the performance didn't change much, or it worsened. I suspect this might change if the training images had been a different size.

For color spaces, I tried using each color space and various numbers of channels. As one might expect, using "ALL" channels tended to perform better than using a single channel, but it also tended to lead to false positives in the test images, suggesting overfitting. I settled on using all the channels of a YCrCb colorspace transform as it seemed to accurately identify the vehicles without many false positives.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the third code cell, I set up some convenience functions for use in fitting the SVC. These functions are used to extract spatial features, color histogram features, and HOG features. In the next code cell, I define a function for extracting single image features and a function for extracting features from a list of images. Towards the bottom of the cell, I set up the parameters I will use and extract the features. I then scale the features using `StandardScaler` and do an 80-20 split on the data to generate training and test data. Finally, near the bottom of the code cell, I fit a standard linear basis function SVC using `LinearSVC`. I repeated this process for various color spaces and choices of channels, spatial features, color features, etc. before arriving at the parameters that remain in the code cell: `YCrCb` color space using all features.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for my sliding windows implementation is in the fifth and sixth code cells. The fifth cell contains convenience functions while the sixth cell implements a 96x96 pixel window with overlap 0.75 run over a test image. You can see that multiple windows detected cars in the image.

The final implementation used in the video is in the last code cell. I started with a window size of 64x64 and searched from the horizon (y = 380 pixels) down to about y = 650 pixels (just above the car hood in the video). When I processed the video, I found that this window size was picking up false positives directly in front of the car, so I reduced the range to 380-550 pixels, basically taking away a row and a half of windows just in front of the car. By reducing this range, it also reduced the x-range, which I changed from 400-1280 pixels to 500-1280 pixels.

Removing the bottom row of windows necessitated adding a set of larger windows. I addded a 96x96 window running over the original range of y = 380-650 and x = 400-1280. This would in theory catch cars directly in front of the camera, though none appear in the image. I did not see any false positives with this size window (after applying heat mapping and thresholding).

Finally, for purposes of heat mapping and thresholding, I added a 128x128 window along the bottom. This added additional "heat" to some of the images so I could be a little more aggressive in ruling out false positives.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I saved the heatmap from one frame to the next and applied a "forgetting factor" to the heatmap so that detected areas would gradually fade away unless they were refreshed with new detections.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11] ![alt text][image12] ![alt text][image13]
![alt text][image14] ![alt text][image15] ![alt text][image16]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image17] ![alt text][image18] ![alt text][image19]
![alt text][image20] ![alt text][image21] ![alt text][image22]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I faced in implementing this project was that I didn't realize for a long time that the VideoFileClip#fl_image function returns an image where each pixel is measured from 0-255 instead of 0-1 like png's. That caused me to get a lot of false positive results!

Once I fixed that bug and got the pipeline working correctly, false positives were less common, but still a problem. The biggest issue I ran into was getting false positives in the road directly in front of the car. In practice, this might cause a control algorithm to slam on the brakes. I used an ad hoc approach and my intuition to remove the false positives because they tended to come from the 64x64 window size, which would typically be much too small a window to enclose a car in that part of the camera view (i.e., directly in front of the car). A more robust way to deal with the problem of false positives would be adding the false positive image as a negative image in the training set.

Because I eliminated the area in front of the car for consideration of sliding windows of 64x64 dimension, I added larger windows (96x96 and 128x128). That negatively impacted the frame rate, and my implementation as is takes around 3 seconds to process each frame. One alternative I did not implement was to compute the HOG features for the image just once and subsample. Another tool would be to reduce the window overlap. I used 0.75 overlap in each dimension, leading to many more windows to compute than lower values like 0.5. I did not aggressively tune this value to reduce computation demand while maintaining adequate identification of cars, but it is easy to see that it has a large influence on the number of windows that must be tested.



