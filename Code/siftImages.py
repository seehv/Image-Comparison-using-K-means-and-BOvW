"""
Created on Sun Oct  4 12:33:32 2020
@author: Harsha Vardhan Seelam

Rescaling image logic explanation:-

* Step 1:- Find aspect ratio of original image

    From original image Aspect Ratio (AR) = width / height

* Step 2:- For target image:-

    Case 1:-
    
        if width < height:-
        the new width after rescaling the input image would be => multiply the aspect ratio (AR) with 480 (target image height)
        the new height after rescaling the input image would be => 480
    
    Case 2:-
    
        if width > height:-
        the new height after rescaling the input image would be => divide 600 (target image width) with aspect ratio (AR)
        the new width after rescaling the input image would be => 600

Hence via this approach, aspect ratio of the original image is preserved.
"""
import cv2 as cv
import numpy as np
import sys
from cv2 import xfeatures2d

#Fucntion for Rescaling the real image while maintaining the aspect ratio which 
#is comparable to an VGA size image 
def rescaleImageWhileMaintainAspectRatio(image, max_height = 480, max_width = 600):
    
    in_ImgH,in_ImgW = image.shape[:2]
    img_AR = in_ImgW/in_ImgH
    
    if in_ImgW < in_ImgH:
        newW = int(img_AR * max_height)
        newH = max_height
    elif in_ImgW > in_ImgH:
        newW = max_width
        newH = int(max_width / img_AR)
        
    return cv.resize(image, (newW,newH), interpolation=cv.INTER_AREA)

#fucntion for extracting the Y-comp of an image
def extractYcomp(image):
    
    YCrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb);
    YCrcb_split = np.array(cv.split(YCrcb));
    return YCrcb_split[0]

#Task -1 fucntion for extracting the keypoints and return a image of circle along with cross inside it
def SIFTImage(Y_comp, resized_img, filename):
    
    #invoking the constructor of SIFT_create() class
    sift = xfeatures2d.SIFT_create()
    keypoints = np.array(sift.detect(Y_comp, None))
    print("# of keypoints in",filename,"is ", len(keypoints))
    
    #insertting a circle a on the image and circle size depends on its radious
    circle_img = cv.drawKeypoints(resized_img, keypoints, resized_img, color = (255,0,0),
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #inserting a cross inside the circled image 
    for keypoint in keypoints:
        #storing location of each keypoint
        x,y = keypoint.pt
        circle_img = cv.drawMarker(resized_img, (int(x),int(y)), (57, 255, 20),
                                   markerType = cv.MARKER_CROSS, 
                                   markerSize= (int(keypoint.size/2)), thickness=1, 
                                   line_type = cv.LINE_AA)
    return circle_img

#fucntion to concatinate the real and keypoints highlighted images
def concatinateImgs(rAR_Img, KpImg):
    
    Final_img = np.concatenate((rAR_Img, KpImg), 1)
    cv.imshow('Task-1 Image with Highlighted Keypoints', Final_img)
    cv.waitKey(0)

#task-2 fucntion to return the keypoints and the descriptors of an given image
def SIFTmulImages(Y_compImages, filename):
    
    sift = xfeatures2d.SIFT_create()
    MulImgKP,MulImgDes = sift.detectAndCompute(Y_compImages, None)
    print("# of keypoints in",filename,"is ", len(MulImgKP))
    return MulImgKP, MulImgDes
 
#Task-2 fucntion to return the luminance Y component and all descriptors of all scaled images  
def Task2ExtractSIFTDesc(in_imgs):
    
    #Temporary variables to store the returns values of the fucntions
    in_MulImgs, rAR_Imgs, Y_compImages, MulImgKP, MulImgDes, allImgDes, TotKeyPts = {},{},{},{},{},[],0
    for i in range(0, len(in_imgs)):
        #read the i image
        in_MulImgs[in_imgs[i]] = cv.imread(in_imgs[i])
        
        #rescale the image
        rAR_Imgs[in_imgs[i]] = rescaleImageWhileMaintainAspectRatio(in_MulImgs[in_imgs[i]])
        
        #extract luminance Y component of the i-th image 
        Y_compImages[in_imgs[i]] = extractYcomp(rAR_Imgs[in_imgs[i]])
        
        #extracting the keypoints and descriptors of the selected image
        MulImgKP[in_imgs[i]], MulImgDes[in_imgs[i]] = SIFTmulImages(Y_compImages[in_imgs[i]], in_imgs[i])
        TotKeyPts = TotKeyPts + len(MulImgKP[in_imgs[i]])
        
        #storing the descriptors for BOW usage
        allImgDes.append(MulImgDes[in_imgs[i]])
    allImgDes = np.concatenate(allImgDes)
    
    return Y_compImages, TotKeyPts, allImgDes

#Fucntion for task-2 to Implementing Bag of Words Algorithm and displaying dissimilarity matrices
def Task2ClusterSIFTDesc(in_imgs, Kvalue, Y_compImages, allImgDes, TotKeyPts):

    detect = cv.xfeatures2d.SIFT_create()
    extract = cv.xfeatures2d.SIFT_create()
    matcher = cv.FlannBasedMatcher()
    
    bowExtracter = cv.BOWImgDescriptorExtractor(extract, matcher)
    # =========================||Implementing K Means algorithm||==================
    # 	clusters count -> dissimilarityPercentage * keypointsTotal / 100
    # =============================================================================
    Kmeans_trainer = cv.BOWKMeansTrainer((int)(Kvalue * TotKeyPts / 100))
    
    #Add keypoint descriptors to the traine
    Kmeans_trainer.add(allImgDes)
    
    #Build vocabulary of visual words and cluster represents a visual word
    bowExtracter.setVocabulary(Kmeans_trainer.cluster())
    
    #Object for storing histogram of the occurrence of the visual words
    img_hist = []
    for i in range(0, len(in_imgs)):
        kps = detect.detect(Y_compImages[in_imgs[i]])
        
        #Computes an image descriptor for visual vocabulary
        bow_desc = bowExtracter.compute(Y_compImages[in_imgs[i]], kps)
        
        #Construct a histogram of the occurrence of the visual words
        img_hist.append(bow_desc)
    img_hist = np.concatenate(img_hist)
    
    #Calculate the Chi-square distance between images
    compHistoramDict = np.zeros((img_hist.shape[0], img_hist.shape[0]),order='F')
    for i in range(0, img_hist.shape[0]):
        for j in range(0, img_hist.shape[0]):
            compHist = cv.compareHist((cv.UMat(img_hist[i])), (cv.UMat(img_hist[j])), cv.HISTCMP_CHISQR)
            compHistoramDict[i][j] = compHist
            # compHistoramDict[in_imgs[i]] = compHist
            # =============================================================================
            #           Algorithm for comparing two histograms using Chi-square distance  
            # 			D(h1,h2) = 2  *  sum_over_all_i [ h1(i) - h2(i) ] ^2  / [ h1(i) + h2(i) ]
            # =============================================================================
    
    #Print dissmilarity matrices for specified K value
    print("=================================================================")
    print("\n"," K = ", Kvalue ," * ",TotKeyPts," = ", (int)(Kvalue * TotKeyPts / 100))
    print("\n"," Dissimilarity Matrix for K= ", Kvalue,"%\n")
    for i in range(0, len(in_imgs)): print("          ", in_imgs[i], end="")
    for i in range(0, compHistoramDict.shape[0]):
        print("\n",in_imgs[i], "     ", end="")
        for j in range(0, compHistoramDict.shape[1]):
            print(round(compHistoramDict[i][j],3),"              ",end="")
            

#==============================||TASK-1||======================================

if (len(sys.argv) < 3):
    
    #read the file
    in_Img = cv.imread(sys.argv[1])
    
    #get rescaled image
    rAR_Img = rescaleImageWhileMaintainAspectRatio(in_Img)
    Real_downScaledImg = np.copy(rAR_Img)
    
    #Extract Luminace of the image
    Y_comp = extractYcomp(rAR_Img)
    
    #get keypoints highlighted image
    kpsHLTD_Img = SIFTImage(Y_comp, rAR_Img, sys.argv[1])
    
    #display image
    concatinateImgs(Real_downScaledImg, kpsHLTD_Img)

#==============================||TASK-2||======================================

elif (len(sys.argv) > 2):
    in_imgs = []
    for i in range(1,len(sys.argv)):
        in_imgs.append(sys.argv[i])
    
    #Get the luminance Y component and all descriptors of all scaled images  
    Y_compImages, TotKeyPts, allImgDes = Task2ExtractSIFTDesc(in_imgs)
    
    #Implementing Bag of Words Algorithm and displaying dissimilarity matrices for K values 5%, 10%, 20%
    Task2ClusterSIFTDesc(in_imgs, 5, Y_compImages, allImgDes, TotKeyPts)
    Task2ClusterSIFTDesc(in_imgs, 10, Y_compImages, allImgDes, TotKeyPts)
    Task2ClusterSIFTDesc(in_imgs, 20, Y_compImages, allImgDes, TotKeyPts)

