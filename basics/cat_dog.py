import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

is_training_set_built = True

def build_training_set():
    new_image_size = 50
    dir = "./kaggle/PetImages/"
    labels = {"Cat": 0, "Dog": 1}
    training_set = []
    cat_count = 0
    dog_count = 0

    for label in labels:
        for file_name in tqdm(os.listdir(dir + label)):
            try:
                path = os.path.join(dir + label, file_name)
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (new_image_size, new_image_size))
                # cv2.imshow(path, image)
                # cv2.waitKey(0)
                training_set.append([np.array(image), np.eye(2)[labels[label]]])

                if label == "Cat":
                    cat_count += 1
                elif label == "Dog":
                    dog_count += 1
            except Exception:
                pass

    np.random.shuffle(training_set)
    np.save("./kaggle/Cat_Dog_Training_Set.npy", training_set)
    print("Total Cat count is: ", cat_count)
    print("Total Dog count is: ", dog_count)
    print("Training set has been built!")
    return training_set

def load_training_set():
    training_set = np.load("./kaggle/Cat_Dog_Training_Set.npy", allow_pickle=True)
    return training_set

training_set = []
if is_training_set_built == False:
    training_set = build_training_set()
else:
    training_set = load_training_set()

plt.imshow(training_set[1][0])
plt.show()