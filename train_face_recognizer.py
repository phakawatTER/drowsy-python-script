import os
import cv2
import numpy as np
import argparse
import pickle



def train_face_model(uid):
    current_directory = str(os.path.dirname(__file__))
    img_dir = os.path.join(current_directory, "training_set",
                        uid)
    x_train = []
    y_labels = []
    ids = {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for index, data in enumerate(os.walk(img_dir), start=0):
        root, dirs, files = data
        if "driver_" not in root:
            continue
        print(index)
        print("__"*20)
        print(root)
        ids["driver_{}".format(index)] = index-1
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                path_to_image = os.path.join(root, file)
                image = cv2.imread(path_to_image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                x_train.append(gray)
                y_labels.append(index)

            


    with open(os.path.join(img_dir, "ids.pickle"), "wb") as file:
        pickle.dump(ids, file)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(os.path.join(img_dir, "trainer.yml"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--uid",required=False,default="")
    args = vars(ap.parse_args())
    train_face_model(args["uid"])
