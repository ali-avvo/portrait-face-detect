import cv2
import sys # Arg Parsing, the easy way
import numpy as np

sys.path.append('utils')
from utils import *

class Image:
    """
    Image Class
    ------------
    Parameters
    :img_path: (default=None), type: str
    -----------
    Methods
    * __len__(): args: None, returns: shape of the image
    * hull(): args: None, returns: hull array
    * face_detect(): args: None, returns: faces array
    * blur(): args: kernel_size=5, returns: portrait bokeh output image
    """
    def __init__(self, img_path=None):
        """
        :param img_path: str, image path, default = None
        """
        if img_path is None:
            print("img_path not mentioned")
        self.path = img_path
        self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        self.img_alpha = cv2.cvtColor(self.img, cv2.COLOR_RGB2RGBA)

        # For face detection
        self.face_cascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
        self.gray = None

    def __len__(self):
        """
        :return: int, number of pixels in self.img object
        """
        return self.img.shape[0] * self.img.shape[1] * self.img.shape[2]

    def hull(self):
        """
        :return: list, list with hull points
        """
        ret, thresh = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hull = []
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
        return hull

    def face_detect(self):
        """
        :return: list, faces detected (points)
        """
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        return faces

    def blur(self, kernel_size=5):
        """
        :param kernel_size: int, kernel size for cv2.GaussianBlur
        :return: image, portrait-bokeh image
        """
        rois = self.face_detect()
        img_cropped = generate_mask(self.img_alpha, rois)

       #blur_image = cv2.GaussianBlur(self.img_alpha, (kernel_size, kernel_size), 0)
        blur_image = cv2.GaussianBlur(self.img_alpha, (kernel_size, kernel_size), 1000, 1000)

        res = overlap(img_cropped, blur_image)
        return res


img_obj = Image(sys.argv[1])
#9 = max blur
portrait_bokeh_image = img_obj.blur(9)
# print("BLUR SHAPE: ", portrait_bokeh_image.shape)
# cv2.imwrite("blur.png", portrait_bokeh_image)

# Ali: What if we create a template blank circular headshot
# of the size we want to show on the profile etc
# Then we overlap our cropped head on that blank circle
# keeping the center (x,y) of the two circles same
# potentially we can fill the empty areas (if any) with blurred part of the headshot
