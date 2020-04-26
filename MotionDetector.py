# MotionDetector.py
# Detects motion on a series of frames.

import cv2, numpy

class MotionDetector:

    def __init__(self, accumulatedWeight=0.5):
        self.accumulatedWeight = accumulatedWeight
        self.bg = None


    # Updates stored background
    def update(self, image):
        # Initialize background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # Update background
        cv2.accumulateWeighted(image, self.bg, self.accumulatedWeight)


    # Detects motion and determine coordinates around area of motion
    def detect(self, image, tolerance=25):
        # Compute the difference between current and previous frame
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        threshold_image = cv2.threshold(delta, tolerance, 255, cv2.THRESH_BINARY)[1]

        # Perform some filtering on the image
        threshold_image = cv2.erode(threshold_image, None, iterations=2)
        threshold_image = cv2.dilate(threshold_image, None, iterations=2)

        # Find contours
        contours = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Version compatibility for cv2
        if len(contours) == 2:
            contours = contours[0]
        else:
            contours = contours[1]

        if len(contours) == 0:
            return None

        (minX, minY) = (numpy.inf, numpy.inf)
        (maxX, maxY) = (-numpy.inf, -numpy.inf)

        for c in contours:
            # compute the bounding box of the contour and use it to
            # update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # Return a tuple of the coordinates of the motion bounding box
        return (minX, minY, maxX, maxY)

