import numpy as np
import cv2
import os
import argparse
import sys
from circular_segmentation import crop_bottom
from glass_circles_Tom import HoughCircles

RESULT_DIRECTORY = "image outputs"
LOW_threshold = 20
HIGH_threshold = 70
threshold = 30
RHO = 1
THETA = np.pi/ 45
LINESTH = 75
DEBUG = True

def DEBUG_SAVE(image, directory):
    """
    Save an image to the result directory for debugging and presenting purposes.
    """
    if DEBUG:
        PATH = f'{RESULT_DIRECTORY}/{directory}'
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        number = len(os.listdir(PATH)) + 1
        cv2.imwrite(f'{PATH}/{number}.jpeg', image)

def canny(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    return edges


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    edges = canny(gray)
    mask = np.zeros_like(edges)
    full_edges = edges.copy()
    inner_circle = cv2.circle(mask, (w//2, h//2), (h//2 - h//8)+1, (255),-1)

    edges = cv2.bitwise_and(edges, inner_circle)

    
    DEBUG_SAVE(full_edges, 'edges')

    lines = cv2.HoughLines(edges, RHO, THETA, LINESTH) 
    result = image.copy()

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

        DEBUG_SAVE(result, 'detected lines')

        return True, len(lines)
    else:
        return False, None

def crop_ridge_band(image):
    inner_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), (image.shape[0]//2 - image.shape[0]//8)+1, (0, 0, 0),-1)
    outer_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), image.shape[0]//2 + 5, (0, 0, 0),10)
    return inner_circle

def full_system(im, offline=False, skip_circle=False):
    if offline:
        im = cv2.imread(im)
    segmented_image, was_segmented = crop_bottom(im)
    if segmented_image is not None:

        DEBUG_SAVE(segmented_image, 'segmented images')

    if was_segmented:
        lines, num = detect_lines(segmented_image)
        if lines and num > 0:
            return "plastic"
        
        else:
            if skip_circle:
                return "glass"
            
            circle_list = HoughCircles(segmented_image)
            if circle_list is not None:
                if len(circle_list) == 0:
                    return "plastic"
                max_r = max(circle_list, key=lambda x: x[2])[2]
                if max_r < 1:
                    return "plastic"
                for i in circle_list:
                    x = i[0] * 3
                    y = i[1] * 3
                    r = i[2] * 3
                    cv2.circle(segmented_image, (x,y), r, (0, 255, 0), 2)
                    
                DEBUG_SAVE(segmented_image, 'detected circles')

                return "glass"
            else:
                return "plastic"
            
    return "unknown"

def process_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: The directory '{directory_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    found = False
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if full_path.lower().endswith(".jpeg"):
            found = True
            classification = full_system(full_path, offline=True)
            print(f"Image '{entry}' is classified as '{classification}'.")

    if not found:
        print(f"No JPEG files found in directory '{directory_path}'.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Classify images of the bottoms of bottles as either plastic or glass.")
    parser.add_argument("mode", choices=['single', 's', 'directory', 'd'],
                        help="Choose 'single' (or 's') to process one image or 'directory' (or 'd') to process all JPEG images in a directory.")
    args = parser.parse_args()

    if args.mode in ['single', 's']:
        image_path = input("Enter the complete path to the image (JPEG): ")
        if not os.path.isfile(image_path) or not image_path.lower().endswith('.jpeg'):
            print("Error: Please provide a valid path to a JPEG image.", file=sys.stderr)
            sys.exit(1)
        classification = full_system(image_path, offline=True)
        print(f"The image is classified as '{classification}'.")
    elif args.mode in ['directory', 'd']:
        directory_path = input("Enter the path to the directory containing JPEG images: ")
        process_directory(directory_path)

if __name__ == '__main__':
    main()