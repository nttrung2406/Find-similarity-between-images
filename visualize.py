from lib import *

def showplt(image, title=None, pltnative=False):
    if pltnative:
        plt.imshow(image)
    else:
        plt.imshow(image[...,::-1])
    plt.title(title)
    plt.xticks([]), plt.yticks([]) 
    plt.show()

def print_similarity(img1, img2, good_matches):
  num_good_matches = len(good_matches)
  print("Similarity:", num_good_matches)

def visualize_similarity(img1, img2, good_matches):
    sift = cv2.SIFT_create()
    keypoints1, _ = sift.detectAndCompute(img1, None)
    keypoints2, _ = sift.detectAndCompute(img2, None)

    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    img_matches[:img1.shape[0], :img1.shape[1]] = img1
    img_matches[:img2.shape[0], img1.shape[1]:] = img2

    # Draw circles around the matched keypoints
    for match in good_matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Offset the second image's keypoints by its width to draw in the correct location
        x2 += img1.shape[1]

        # Draw a circle around the matched keypoints
        cv2.circle(img_matches, (int(x1), int(y1)), 4, (0, 255, 0), 1)
        cv2.circle(img_matches, (int(x2), int(y2)), 4, (0, 255, 0), 1)

    cv2.imshow("Matches", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
