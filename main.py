def main():
  img1 = cv2.imread("/content/Image/2.jpg")
  img2 = cv2.imread("/content/Image/3.jpg")

  sift = cv2.SIFT_create()
  keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
  keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

  flann = cv2.FlannBasedMatcher()
  matches = flann.knnMatch(descriptors1, descriptors2, k=2)

  good_matches = []
  for m1, m2 in matches:
    if m1.distance < 0.8 * m2.distance:
      good_matches.append(m1)
  similarity_percentage = calculate_similarity_percentage(img1, img2)

  print("Similarity Percentage: {:.2f}%".format(similarity_percentage))
  print_similarity(img1, img2, good_matches)
  visualize_similarity(img1, img2, good_matches)

if __name__ == "__main__":
  main()
