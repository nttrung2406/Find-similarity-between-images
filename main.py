from lib import *
from visualize import *

def main():
    img1 = cv2.imread("C:/Users/flori/Downloads/vts1.jpg")
    img2 = cv2.imread("C:/Users/flori/Downloads/vts2.jpg")

    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return

    img1_resized = cv2.resize(img1, (600, 600))
    img2_resized = cv2.resize(img2, (600, 600))

    size_img1 = img1_resized.shape[:2]  
    size_img2 = img2_resized.shape[:2] 

    print("Size of img1:", size_img1)
    print("Size of img2:", size_img2)

    gray_img1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_img2, None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.99 * m2.distance:
            good_matches.append(m1)

    similarity_percentage = len(good_matches) / min(len(keypoints1), len(keypoints2)) * 100.0


    print("Similarity Percentage: {:.2f}%".format(similarity_percentage))
    print_similarity(gray_img1, gray_img2, good_matches)
    visualize_similarity(img1_resized, img2_resized, good_matches)

if __name__ == "__main__":
    main()

