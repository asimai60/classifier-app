import numpy as np
import cv2
import os
from circular_segmentation import crop_bottom



plastic_path = 'segmented bottoms/separation/plastic'
glass_path = 'segmented bottoms/separation/glass'
LOW_threshold = 20
HIGH_threshold = 70
threshold = 30
RHO = 1
THETA = np.pi/ 45
LINESTH = 75

def load_and_resize(path):
    image = cv2.imread(path)
    desired_shape = (480, 480)
    image = cv2.resize(image, desired_shape)
    return image

def canny(gray):
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    return edges


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    crop_size = 1/8  
    x_start = int(w * crop_size)
    x_end = int(w * (1 - crop_size))
    y_start = int(h * crop_size)
    y_end = int(h * (1 - crop_size))

    gray_cropped = gray[y_start:y_end, x_start:x_end]
    edges = canny(gray_cropped)
    edges_full = canny(gray)


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

        # cv2.imshow('Original Image', edges)
        # cv2.imshow('Detected Lines', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        number = len(os.listdir('server/detected lines')) + 1
        PATH = 'server/detected lines'
        cv2.imwrite(f'{PATH}/detected{number}.jpeg', result)
        return True, len(lines)
    else:
        #print("no lines detected")

        # cv2.imshow('Original Image', edges_full)
        # cv2.imshow('Detected Lines', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return False, None
    
def extract_ridges(PATH, im, image=None):
    image = cv2.imread(f'{PATH}{im}') if image is None else image
    original = image.copy()
    inner_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), image.shape[0]//2 - image.shape[0]//8, (0, 0, 0),-1)
    gray_image = cv2.cvtColor(inner_circle, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (9,9), 0)
    LoG = cv2.Laplacian(denoised_image, cv2.CV_64F, ksize=5)
    # cv2.imshow(f'LoG {im.split(".")[0]}', LoG)
    # cv2.imshow(f'Original {im.split(".")[0]}', original)
    # cv2.imwrite(f'ridge patterns/{im.split(".")[0]}_ridge.jpeg', LoG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return LoG

def crop_ridge_band(image):
    inner_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), (image.shape[0]//2 - image.shape[0]//8)+1, (0, 0, 0),-1)
    outer_circle = cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), image.shape[0]//2 + 5, (0, 0, 0),10)
    return inner_circle


def compare_patterns(target_img, pattern_img):
    sift = cv2.SIFT_create(edgeThreshold=10, nfeatures=1000, nOctaveLayers=3, contrastThreshold=0.04, sigma=1.6)

    pattern_kp, pattern_des = sift.detectAndCompute(pattern_img, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    target_kp, target_des = sift.detectAndCompute(target_img, None)

    matches = flann.knnMatch(pattern_des, target_des, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    score = len(good_matches)
    print(f"Similarity Score: {score}")
    return score

def test(path, label):
    files = os.listdir(path)
    amount = len(files)
    correct = 0                                                                                
    for image_file in files:
        image_path = os.path.join(path, image_file)
        image = load_and_resize(image_path)
        lines = detect_lines(image)
        correct += (lines and label == "plastic") or (not lines and label == "glass")

    return (amount - correct)/amount * 100, correct/amount * 100

def template_ridges():
    segmented, _ = crop_bottom('template.jpeg', 'bottom bottles/')
    ridge = extract_ridges('', '', segmented)
    cropped_template = crop_ridge_band(ridge)
    return cropped_template

def compare_ridges(PATH, im, segmented_image):
    ridge_image = extract_ridges(PATH, im, segmented_image)
    cropped_ridge = crop_ridge_band(ridge_image)
    cv2.imwrite('cropped_ridge.jpeg', cropped_ridge)

    if not os.path.exists('template_ridge.jpeg'):
        cv2.imwrite('template_ridge.jpeg', template_ridges())
    cropped_ridge = cv2.imread('cropped_ridge.jpeg', cv2.IMREAD_GRAYSCALE)
    template_ridge = cv2.imread('template_ridge.jpeg', cv2.IMREAD_GRAYSCALE)
    score = compare_patterns(cropped_ridge, template_ridge)
    return score

def full_system(im):
    segmented_image, was_segmented = crop_bottom(im)
    if segmented_image is not None:
        print(os.getcwd())
        SAVE_PATH = 'server/OCI'
        number = len(os.listdir(SAVE_PATH)) + 1
        cv2.imwrite(f'{SAVE_PATH}/cropped{number}.jpeg', segmented_image)
    if was_segmented:
        lines, num = detect_lines(segmented_image)
        if lines and num > 0:
            print(f"Detected {num} lines")
            return "plastic"
        else:
            return "glass"
            score = compare_ridges(PATH, image, segmented_image)
            if score > 10:
                return "glass"
            return "plastic without lines"
    return "unknown"

def main():
    return 0

if __name__ == '__main__':
    main()