# Methods for finding similarity between images with ML and Non-ML application

**Normalized Cross Correlation NCC**

_ Slide the "template" (smaller image) on top of the larger image. 

_ Calculate NCC for each position. 

_ The location with the highest NCC score may be the location with the most similar region in the larger image.

Time execute: 2.5s

Accuracy: 50%

**Cosine similarity using CLIP model**

_ Use CLIP to convert from image to digital matrix.

_ Use cosine similarity to find the distance between images.

Time execute: 0.93s

Accuracy: 90%

**SIFT**

_ SIFT will detect keypoints in images and compare those pixels between images

_ The downside is that you can set the threshold yourself so it may not be accurate (the current threshold is 0.988 for the 90% threshold).

Time execute: 27.2s

Accuracy: 70%



