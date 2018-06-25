# PillView
Testing ground for improving medication recognition with minimal data

### 1) Impact of image manipulation
Using pre-trained vgg-16 with Tensorflow. Training set of 40, Test set of 10 and Validation set of 10.

#### Test Image
![](images/test.png) 

#### Image manipulation included 4 different manipulation algorithms:
##### Original

![](images/capsule_1_resize.jpg "Original")![](images/color_resize.jpg "Original")![](images/hist_org_resize.jpg "Histogram")

##### Sharpen

![](images/capsule_1_s.jpg "Sharpen")![](images/sharpen.png "Sharpen")![](images/hist_sharpen.jpg "Histogram")

##### Gaussian

![](images/capsule_1_g.jpg "Gaussian")![](images/gausian.png "Gaussian")![](images/hist_guasian.jpg "Histogram")

##### Invert

![](images/capsule_1i.jpg "Invert")![](images/invert.png "Invert")![](images/hist_invert.jpg "Histogram")

##### Edges

![](images/capsule_1_e.jpg "Original")![](images/edges.png "Sharpen")![](images/hist_edge.jpg "Histogram")

### 2) Effect of different transformations on accuracy and error rate

 - Experimention done using transforms_playground.ipynb
 
 - The train set has 1040 images and the test set has 208 images
 - Training was done using Leslie Smith’s one cycle method
 - Training was done with a high learning rate, low cycle length, dedicating 5% of the cycle to annealing at the end with a maximum 
   momentum of 0.95 and a minimum momentum of 0.85 and a weight decay of 1e-5
 - There were a total number of 82 different labels that ranged from describing the shape, color, surface markings, strength; for  
   example a capsule could have the following labels [capsule], [blue], [TEVA], [multi color], [white], [capsule shape], [25mg] etc
 - The results were broken down into 1) N: The total number of correct labels predicted, 2) Accuracy: (N/82)*100 where 82 is the total
   number of labels, 3) E: The total number of incorrect labels predicted and 4) Error: (E/N)*100
   
#### Outcomes

The aim of this project was to find the highest accuracy with the lowest number of errors in the shortest amount of time (average training time per 10 epochs was around 3 minutes) we can see that by using various data augmentation techniques plus Leslie Smith’s one cycle approach we can get a good indication of what data augmentation techniques work best for this data set.

![](images/transform_result.png "Transformation Results")

The summary of the results shows that [RandomZoomRotate] had the most correctly chosen labels and [Padding(50)] had the least number of incorrectly chosen labels. Vice versa [Cutout] had the least number of correctly chosen labels and [RandomLighting + Dihedral] had the highest number of incorrectly chosen numbers

![](images/transforms_total.png "Transformation Results")

 

