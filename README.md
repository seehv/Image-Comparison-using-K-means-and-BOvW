# Image-Comparison-using-K-means-and-BOvW
SIFT Descriptor implementation has been used in this program to find the similarity between the images 

#### About the program
In this program, the flow has been divided into 2 tasks

Task-1: 
+ Can process only single image
+ initially the image will be rescaled into VGA compatible size to redce the computation
+ Nect we will extract the Y-component (aka luminance) pannel from the image and we will use it to find the keyponints in an image. 
+ Will show the Number of keypoint found in an image.
How to run the program using cmd for task-1? 
  ###### /...> siftImages imagefile1

Task-2:
+ Can process multiple images
+ For each image, same process as task-1
+ save all these deriptors in a single vector and create a Bag-of-Visual-Words
+ Use the BoVW to train our kmeans trainner
+ Now again for each image, find the decriptors. 
+ use those decriptors found in the previous step to search in the k-maenas model.
+ display the Display the Dissimilarity matrix for all the images given in the CMD.
How to run the program using cmd for task-2? 
  ###### /...> siftImages imagefile1 imagefile2 imagefile3 imagefile4...
