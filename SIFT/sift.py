import sys
sys.path.append("C:/Users/flori/OneDrive/Máy tính/Find-similarity-between-images-main") #folder path
from lib import *

MIN_MATCH_COUNT = 10
def compare_images(img1_path, img2_path):

  img1 = cv2.imread(img1_path, 0)  
  img2 = cv2.imread(img2_path, 0)

  sift = cv2.SIFT_create()

  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)

  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1, des2, k=2)

  good = []
  for m, n in matches:
    if m.distance < 0.988 * n.distance:
      good.append(m)

  similarity_ratio = len(good) / len(matches) if len(matches) > 0 else 0
  similarity_percentage = min(similarity_ratio * 100, 100)


  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  
                       singlePointColor=None,
                       matchesMask=matchesMask,  
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img3, 'gray'), plt.show()
    print(f"Image pair: {img1_path} - {img2_path}")
    print(f"Similarity percentage: {similarity_percentage:.2f}%")

  else:
    print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT} for {img1_path} - {img2_path}")


def main():
  image_folder = "C:/Users/flori/Downloads/image"
  image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
                 if filename.endswith(('.jpg', '.png', '.jpeg'))]
  start = time.time()
  for i in range(len(image_paths)):
    for j in range(i + 1, len(image_paths)):
      compare_images(image_paths[i], image_paths[j])
  end = time.time()
  print(f"Total time taken: {round((end - start), 2)} s")

if __name__ == "__main__":
    main()
