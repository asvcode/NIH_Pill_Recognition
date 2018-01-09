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

[Fastai repository](https://github.com/fastai/fastai) Fastai Version 2 Part 1 repository

# Lesson Notes
I have included a compilaton of my notes and understanding below:

Lesson 1 Cats and Dogs notes:3  [Fastai Forum](http://forums.fast.ai/t/cats-and-dogs-code-notes/7561) | [PDF Link](images/lesson1_notes.pdf "PDF Link")

<p align="center">
<imgsrc="images/lesson1_notes_Page_01.jpg" width=110/><img src="images/lesson1_notes_Page_02.jpg" width=110 /><img src="images/lesson1_notes_Page_03.jpg" width=110 /><img src="images/lesson1_notes_Page_04.jpg" width=110 /><img src="images/lesson1_notes_Page_05.jpg" width=110 /><img src="images/lesson1_notes_Page_06.jpg" width=110 /><img src="images/lesson1_notes_Page_07.jpg" width=110 /><img src="images/lesson1_notes_Page_08.jpg" width=110 />
</p>
