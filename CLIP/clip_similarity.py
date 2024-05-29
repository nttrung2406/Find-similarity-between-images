from sentence_transformers import SentenceTransformer, util
import sys
sys.path.append("C:/Users/flori/OneDrive/Máy tính/Find-similarity-between-images-main")
from lib import *

print('Loading CLIP Model...')
model = SentenceTransformer('clip-ViT-B-32')

image_folder = "C:/Users/flori/Downloads/image"
image_paths = []

for file_path in glob.glob(os.path.join(image_folder, '*')):
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(file_path)

print("Images:", len(image_paths))

encoded_images = []
for filepath in image_paths:    
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Unable to load image '{filepath}'")
    else:
        encoded_images.append(img)

print("Images loaded:", len(encoded_images))
encoded_image = model.encode([Image.open(filepath) for filepath in image_paths], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

processed_images = util.paraphrase_mining_embeddings(encoded_image) #cosine similarity
NUM_SIMILAR_IMAGES = 10 

print('Finding duplicate images...')
start_dupl = time.time()
duplicates = [image for image in processed_images if image[0] >= 1]
for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_paths[image_id1])
    print(image_paths[image_id2])
end_dupl = time.time()
print("Duplicate search spent: ", end_dupl - start_dupl)

print('Finding near duplicate images...')
start_2 = time.time()
threshold = 0.94
near_duplicates = [image for image in processed_images]
filtered_near_duplicates = [image for image in processed_images if image[0] > threshold]

# for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
#     print("\nScore: {:.3f}%".format(score * 100))
#     print(image_paths[image_id1])
#     print(image_paths[image_id2])
#     img1 = cv2.imread(image_paths[image_id1])
#     img2 = cv2.imread(image_paths[image_id2])


for score, image_id1, image_id2 in filtered_near_duplicates[0:NUM_SIMILAR_IMAGES]:
    print("\nScore: {:.3f}%".format(score * 100))
    print(image_paths[image_id1])
    print(image_paths[image_id2])
    img1 = cv2.imread(image_paths[image_id1])
    img2 = cv2.imread(image_paths[image_id2])
end_2 = time.time()
print("Near duplicate search spent: ", end_2 - start_2, " s")
