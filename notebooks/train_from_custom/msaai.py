# import modules
import os  # file
import cv2
import shutil
import random
from numpy import split
import pandas as pd


# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [
        base_colors[color_index][i]
        + increments[color_index][i] * (cls_num // len(base_colors)) % 256
        for i in range(3)
    ]
    return tuple(color)


def drawPredictionBox(img, class_name, box):
    # get coordinates
    [x1, y1, x2, y2] = box.xyxy[0]
    # convert to int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    vertex1 = (x1, y1)
    vertex2 = (x2, y2)

    confidence = box.conf[0]
    colour = getColours(int(box.cls[0]))

    # draw prediction retangle
    cv2.rectangle(img, vertex1, vertex2, colour, 2)

    # put the class name and confidence on the image
    cv2.putText(
        img,
        f"{class_name} {confidence:.2f}",
        vertex1,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        colour,
        2,
    )
    return img


def drawPredictions(predictions, img, class_names, treshold=0.4):
    summary = ""
    for p in predictions:
        summary = p.verbose()
        boxes = p.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence > treshold:
                cls = int(box.cls[0])  # get the class index
                # get the class name
                class_name = class_names[cls]
                drawPredictionBox(img, class_name, box)
    return summary


# create directories to organize images and cleanup for a new to avoid duplicate images
def reset_directories(directories):
    """
    Check if the specified directories exist. If they do, delete them and recreate them.
    Ensures the directories are clean before use.

    Parameters:
        directories (list): List of directories to reset.
    """
    for path in directories:
        if os.path.exists(path):
            # delete the directory and all its contents
            try:
                shutil.rmtree(path)
                print(f"Deleted existing directory: {path}")
            except Exception as e:
                print(f"Failed to delete {path}. Reason: {e}")

        # Recreate the directory
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Recreated directory: {path}")
        except Exception as e:
            print(f"Failed to create directory {path}. Reason: {e}")


# SPLIT IMAGE DATASET AND FOLDERS
def arrange_image_and_label_files(
    df, source_image_dir, source_label_dir, image_dirs, label_dirs
):
    for _, row in df.iterrows():
        img_file = row["filename"]
        split = row["split"]

        # Source paths
        img_src = os.path.join(source_image_dir, img_file)
        label_src = os.path.join(
            source_label_dir, os.path.splitext(img_file)[0] + ".txt"
        )

        # Destination paths
        img_dest = os.path.join(image_dirs[split], img_file)
        label_dest = os.path.join(
            label_dirs[split], os.path.splitext(img_file)[0] + ".txt"
        )

        # Copy image file
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dest)

        # Copy corresponding label file
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dest)


def img_train_test_split(source_image_dir):
    """
    Base on files in source_image_dir, split images following rule
    Parameters:
        source_image_dir (str): Original Images Directory to split
    """
    # Set random seed for reproducibility
    random.seed(42)

    # extract the image files
    image_files = [
        f for f in os.listdir(source_image_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(image_files)

    # Create DataFrame with file paths and dataset split assignments
    split_files = pd.DataFrame({"filename": image_files})

    # dynamically split dataset into train, valid, and test
    # split sizes
    train_size = int(0.7 * len(split_files))  # 70% for training
    valid_size = int(0.2 * len(split_files))  # 20% for validation
    test_size = len(split_files) - train_size - valid_size  # remaining 10% for testing

    # split labels
    train_labels = ["train"] * train_size
    valid_labels = ["valid"] * valid_size
    test_labels = ["test"] * test_size

    split_files["split"] = train_labels + valid_labels + test_labels
    return split_files


def setup_image_directories(base_dir):
    return
