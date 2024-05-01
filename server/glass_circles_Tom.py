import numpy as np
import cv2
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter

LOW_threshold = 20
HIGH_threshold = 70
threshold = 30
RHO = 1
THETA = np.pi/ 45
LINESTH = 75

Local_max_Th = 0.6
LOW2_threshold = 30
HIGH2_threshold = 150
bin_size = 3


def load_and_resize(path):
    image = cv2.imread(path)
    desired_shape = (480, 480)
    image = cv2.resize(image, desired_shape)
    return image


def canny(gray):
    # Apply Gaussian Blurring to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, LOW_threshold, HIGH_threshold)
    return edges


def detect_lines(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    # Apply edge detection using the Canny edge detector
    edges = canny(gray)
    mask = np.zeros_like(edges)
    inner_circle = cv2.circle(mask, (w//2, h//2), (h//2 - h//8)+1, (255),-1)
    edges = cv2.bitwise_and(edges, inner_circle)
    # Use HoughLines to detect lines in the edge map
    lines = cv2.HoughLines(edges, RHO, THETA, LINESTH)  # These parameters may need adjustment for your specific case
    # Create a copy of the original image to draw lines on
    result = image.copy()
    return lines is not None


def canny2(gray):
    # Apply Gaussian Blurring to reduce noise and improve edge detection
    center = gray.shape[0]//2
    inner_circle = cv2.circle(gray, (center, center), center - gray.shape[0]//8, (0, 0, 0),-1)
    blurred = cv2.GaussianBlur(inner_circle, (5, 5), 0)
    # Use adaptive thresholding to create a binary image
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(adaptive_thresh, LOW2_threshold, HIGH2_threshold)
    return edges


def generate_accumulator(edge_map, max_radius, bin_size=1):
    # Use uniform_filter to effectively "bin" the image
    # by computing the local mean over the areas of size (bin_size x bin_size)
    # This is much more efficient than manually summing up the pixels
    if bin_size > 1:
        edge_map = uniform_filter(edge_map, size=bin_size)
        edge_map = edge_map[::bin_size, ::bin_size]

    # Initialize the accumulator array for the binned edge map
    accumulator = np.zeros((max_radius, *edge_map.shape))

    # Create a two-dimensional grid of coordinates for the binned edge map
    y, x = np.indices(edge_map.shape)
    max_radius_binned = max_radius // bin_size + 1
    # Generate circle masks and accumulate
    for radius in range(1, max_radius_binned):
        # Generate the mask for this radius
        mask = np.zeros_like(edge_map, dtype=float)
        # rr, cc = circle_perimeter(edge_map.shape[0]//2, edge_map.shape[1]//2, radius)
        

        #for pyinstaller
        height, width = edge_map.shape[:2]
        image = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        cv2.circle(image, center, radius, (255), 1, lineType=cv2.LINE_8)
        rr, cc = np.where(image > 0)
        ########

        # Ensure that the indices are within the dimensions of 'mask'
        rr = np.clip(rr, 0, mask.shape[0] - 1)
        cc = np.clip(cc, 0, mask.shape[1] - 1)

        mask[rr, cc] = 1

        # Convolve and accumulate the results in the corresponding layer of the accumulator
        accumulator[radius - 1] = fftconvolve(edge_map, mask, mode='same')

    return accumulator


def find_local_maxima(accumulator, threshold=0.5, neighborhood_size=3):
    threshold_abs = threshold * np.max(accumulator)
    local_maxima = np.zeros_like(accumulator, dtype=bool)

    # Iterate over each pixel in the accumulator
    for index, value in np.ndenumerate(accumulator):
        if value >= threshold_abs:
            # Define the neighborhood boundaries
            min_bound = np.maximum(np.subtract(index, (neighborhood_size // 2,)*len(index)), 0)
            max_bound = np.minimum(np.add(index, (neighborhood_size // 2 + 1,)*len(index)), accumulator.shape)

            neighborhood = accumulator[min_bound[0]:max_bound[0], min_bound[1]:max_bound[1]]

            # Check if the current pixel is the maximum within its neighborhood
            if value == np.max(neighborhood):
                local_maxima[index] = True
    peaks_indices = np.argwhere(local_maxima)
    return peaks_indices


def HoughCircles(image):
    # Set the radius, cx and cy range
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = canny2(gray)
    height, width = edge_map.shape
    max_radius = width // 2

    accumulator = generate_accumulator(edge_map, max_radius, bin_size)
    local_maxima = find_local_maxima(accumulator, Local_max_Th)
    # Create a condition where the second and third columns are not both 81
    mask = ~(local_maxima[:, 1] == 81) | ~(local_maxima[:, 2] == 81)

    # Apply the mask to 'local_maxima' to filter out the unwanted row
    local_maxima = local_maxima[mask]
    return local_maxima