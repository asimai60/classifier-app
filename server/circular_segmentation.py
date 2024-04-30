import numpy as np
import cv2
import os

def load_image(path, max_size=800):
    image = cv2.imread(path)
    # factor = int(np.round(max(image.shape[0], image.shape[1]) / max_size))
    # desired_shape = (image.shape[1]//factor, image.shape[0]//factor) if factor > 1 else image.shape[:2]
    # image = cv2.resize(image, desired_shape)
    return image

def detect_circles(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (5,5), 0)
    edges = cv2.Canny(denoised_image, 20, 0)

    dp = 1
    min_dist = 200
    param1 = 70
    param2 = 70
    min_radius = image.shape[0] // 8
    max_radius = 0

    hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if hough_circles is None:
        param1 = 50
        param2 = 50
        hough_circles = cv2.HoughCircles(denoised_image, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return hough_circles

def radius_smaller_than_half(center, radius, image_shape):
    a = center[0].astype(int) - radius.astype(int)
    b = center[1].astype(int) - radius.astype(int)
    c = center[0].astype(int) + radius.astype(int)
    d = center[1].astype(int) + radius.astype(int)
    if a < 0 or b < 0 or c >= image_shape[1] or d >= image_shape[0]:
        return False                        
    return True


def segment_circles(image, hough_circles):
    mask = 255 * np.ones_like(image)
    x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]
    if hough_circles is not None:
        n = 2
        for iter in range(n):
            mask = np.zeros_like(image)
            hough_circles = np.uint16(np.around(hough_circles))
            max_radius_index = np.argmax(hough_circles[0, :, 2])
            i = hough_circles[0, max_radius_index]
            center = (i[0], i[1])
            radius = i[2]
            if  radius_smaller_than_half(center, radius, image.shape): break
            hough_circles = np.delete(hough_circles, max_radius_index, axis=1)
            if iter == n-1:
                return image, False
            
        x, y = center
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
    else:
        return image, False
    
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = image[y1:y2, x1:x2] if x1 < x2 and y1 < y2 else image

    return image, True

def crop_bottom(image):
    hough_circles = detect_circles(image)
    segmented_image, was_segmented = segment_circles(image, hough_circles)
    standard_size = (480, 480)
    segmented_image = cv2.resize(segmented_image, standard_size)
    return segmented_image, was_segmented

def main():
    PATH = 'bottom bottles/'
    images = os.listdir(PATH)
    images.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else np.inf)
    for im in images:
        segmented_image, was_segmented = crop_bottom(im, PATH)
        if was_segmented:
            # cv2.imshow(f'Detected Circles in {im}', segmented_image)
            cv2.imwrite(f'segmented bottoms/{im}', segmented_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print(f'No circles detected in {im}')

if __name__ == '__main__':
    main()
