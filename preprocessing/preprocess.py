import cv2

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    def gauss_img_sharpening(self, img):
        imgG = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.addWeighted(img, 1.5, imgG, -0.5, 0)
        return img

    def invert(self, img):
        img = (255 - img)
        return img