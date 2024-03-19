# Project: Panoramic View
# Author: Guillem Bibiloni Femenias

import numpy as np
import cv2
import math
import sys

from numpy import ndarray
from typing import List

"""
Function   : computePairPanoramic
Description: Given a two sequential images it joins them together (image stitching).
Parameters :
            Mandatory    
              - img1: Left image. Type: ndarray.
              - img2: Right image. Type: ndarray.

            Optional
              - match_max_width_1: Bound limit in which matches are searched in the left image. To prevent
                                   computing homography matrix with matches that do not correspond to sequential
                                   images. Type: int.

              - match_max_width_2: Bound limit in which matches are searched in the right image. To prevent
                                   computing homography matrix with matches that do not correspond to sequential
                                   images. Type: int.

              - scale_factor: Factor in which the images are resized in order to compute the operations (SIFT,
                              BFmatcher, Homography finding...) that build the panoramic image. Range between
                              (0,1], if None the scale_factor is set authomatically. Type: Float.

              - p_best_matches: Percentage of matches that will be used in finding the homography matrices.
                                Range (0,1], if 1 all the matches will be used. Type: Float.

              - cut_black_part: Indicates wether if the leftover black parts of the image, when warp perspective
                                is applied, are removed or not. Type: Boolean.

              - ransac_reproj_threshold: it is a hyperparameter for finding the homography matrix using the RANSAC method.
                                         A lower value restricts more the algorithm considering only very close matches as
                                         not outliers, and a higher value makes the algorithm more permissive (recommended
                                         with scenes with larger variations between points and more noise). Type: float.

Return     : The result (joined) stitched image. Type: ndarray.
"""
def computePairPanoramic(   img1: ndarray,
                            img2: ndarray,
                            match_max_width_1=(sys.maxsize - 1),
                            match_max_width_2=(sys.maxsize - 1),
                            scale_factor=None,
                            p_best_matches=1,
                            cut_black_part=True,
                            ransac_reproj_threshold=50.0) -> ndarray:
    
    
    # Convert to grayscale (only if they are colored images)
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2


    # Resize images
    scale_factor_1 = scale_factor
    scale_factor_2 = scale_factor
    # If scale_factor is None the biggest dimension is set to
    # 1000 and the other resized to mantain aspect ratio.
    if scale_factor is None:
        MAX_SIZE = 1000
        if np.max(gray1.shape) > MAX_SIZE:
            scale_factor_1 = MAX_SIZE / np.max(gray1.shape)
        else:
            scale_factor_1 = 1

        if np.max(gray2.shape) > MAX_SIZE:
            scale_factor_2 = MAX_SIZE / np.max(gray2.shape)
        else:
            scale_factor_2 = 1

        
    gray1 = cv2.resize(gray1, None, fx=scale_factor_1, fy=scale_factor_1)
    gray2 = cv2.resize(gray2, None, fx=scale_factor_2, fy=scale_factor_2)
    
    

    # Keypoint region selection, 
    # we only need to search within the region corresponding to the last image added on each side
    mask1 = np.zeros_like(gray1)
    mask1[:,-math.floor(match_max_width_1 * scale_factor_1):] = 255

    mask2 = np.zeros_like(gray2)
    mask2[:,:math.floor(match_max_width_2 * scale_factor_2)] = 255


    #Create the SIFT object
    sift = cv2.SIFT_create()

    # Detect the keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)
    
    # Compute the matches (by brute force)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
        
    good_matches = matches
    # Get the best matches if needed
    if p_best_matches != 1:
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        n_best_matches = math.floor(len(good_matches) * p_best_matches)
        good_matches = good_matches[:n_best_matches]

    # Prepare the source and destination matching points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Return the points to its original size
    src_pts = src_pts / scale_factor_1 
    dst_pts = dst_pts / scale_factor_2 
    
    # Calculate homography matrix
    homography, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_reproj_threshold)
    
    # Using the homography matrix transform the second image (warping its perspective)
    result = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    
    # Copy the first image into the result image.
    result[0:img1.shape[0], 0:img1.shape[1]] = img1


    # Cut the left black part if needed
    if cut_black_part:
        if len(result.shape) == 3:
            gray_r = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray_r = result
        non_black_coords = cv2.findNonZero(gray_r)
        last_pixel_pos = np.max(non_black_coords[:, 0, 0])
        return result[:,:last_pixel_pos],img1.shape[1]
    else:
        return result,img1.shape[1]
    

"""
Function   : computePanoramicView
Description: Given a list of sequential images it constructs a panoramic view joining them together. It uses
             computerPairPanoramic function.
Parameters :
            Mandatory    
              - img_list: A list of images (of type ndarray) that are sequential, from left to right. Being [0]
                          pos the leftmost image and [len - 1] pos the rightmost image. Type: List.

            Optional
              - scale_factor: Factor in which the images are resized in order to compute the operations (SIFT,
                              BFmatcher, Homography finding...) that build the panoramic image. Range between
                              (0,1], if None the scale_factor is set authomatically. Type: Float.

              - p_best_matches: Percentage of matches that will be used in finding the homography matrices.
                                Range (0,1], if 1 all the matches will be used. Type: Float.

              - cut_black_part: Indicates wether if the leftover black parts of the image, when warp perspective
                                is applied, are removed or not. Type: Boolean.
            
              - ransac_reproj_threshold: it is a hyperparameter for finding the homography matrix using the RANSAC method.
                                         A lower value restricts more the algorithm considering only very close matches as
                                         not outliers, and a higher value makes the algorithm more permissive (recommended
                                         with scenes with larger variations between points and more noise). Type: float.

Return     : The result panoramic image. Type: ndarray.
"""
def computePanoramicView(   img_list: List,
                            scale_factor=None,
                            p_best_matches=1,
                            cut_black_part=True,
                            ransac_reproj_threshold=50.0) -> ndarray:
    
    # Process left part
    result_left = img_list[0]
    match_max_width_left = sys.maxsize - 1
    for i in range(1,len(img_list) // 2 + 1):
        # Flip input images
        mirr_img_i = cv2.flip(img_list[i],1)
        result_left = cv2.flip(result_left,1)
        result_left,match_max_width_left = computePairPanoramic(mirr_img_i,
                                              result_left,
                                              match_max_width_2=match_max_width_left,
                                              scale_factor=scale_factor,
                                              p_best_matches=p_best_matches,
                                              cut_black_part=cut_black_part,
                                              ransac_reproj_threshold=ransac_reproj_threshold)
        # Flip result
        result_left = cv2.flip(result_left,1)

    
    # Process right
    result_right = img_list[-1]
    match_max_width_right = sys.maxsize - 1
    for i in range(len(img_list) - 2,len(img_list) // 2,-1):
        result_right,match_max_width_right = computePairPanoramic(img_list[i],
                                          result_right,
                                          match_max_width_2=match_max_width_right,
                                          scale_factor=scale_factor,
                                          p_best_matches=p_best_matches,
                                          cut_black_part=cut_black_part,
                                          ransac_reproj_threshold=ransac_reproj_threshold)

    # Join two parts
    result,_ = computePairPanoramic(result_left,
                                    result_right,
                                    match_max_width_1=match_max_width_left,
                                    match_max_width_2=match_max_width_right,
                                    scale_factor=scale_factor,
                                    p_best_matches=p_best_matches,
                                    cut_black_part=cut_black_part,
                                    ransac_reproj_threshold=ransac_reproj_threshold)
    return result



