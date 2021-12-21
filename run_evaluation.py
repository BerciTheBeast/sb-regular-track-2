import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']
        self.config = config

    def get_annotations(self, annot_name):
            with open(annot_name) as f:
                lines = f.readlines()
                annot = []
                for line in lines:
                    l_arr = line.split(" ")[1:5]
                    l_arr = [float(i) for i in l_arr]
                    annot.append(l_arr)
            return annot
    
    def run_preprocessing(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        preprocess = Preprocess()
        
        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            filename = str.split(im_name, "\\")[1]
            # cv2.imshow('orig', img)
            # cv2.waitKey(0)
            
            # Apply some preprocessing
            # images_hist_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            # cv2.imshow('images_hist_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_hist_path'] + "/" + filename, img)

            # images_gauss_path
            img = cv2.imread(im_name)
            img = preprocess.gauss_img_sharpening(img)
            # cv2.imshow('images_gauss_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_gauss_path'] + "/" + filename, img)

            # images_hist_gauss_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.gauss_img_sharpening(img)
            # cv2.imshow('images_hist_gauss_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_hist_gauss_path'] + "/" + filename, img)

            # images_invert_path
            img = cv2.imread(im_name)
            img = preprocess.invert(img)
            # cv2.imshow('images_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_invert_path'] + "/" + filename, img)

            # images_hist_invert_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_hist_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_hist_invert_path'] + "/" + filename, img)

            # images_gauss_invert_path
            img = cv2.imread(im_name)
            img = preprocess.gauss_img_sharpening(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_gauss_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_gauss_invert_path'] + "/" + filename, img)

            # images_hist_gauss_invert_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.gauss_img_sharpening(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_hist_gauss_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(self.config['images_hist_gauss_invert_path'] + "/" + filename, img)

        im_list = sorted(glob.glob(str.replace(self.images_path, 'test', 'train') + '/*.png', recursive=True))
        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
            filename = str.split(im_name, "\\")[1]
            # cv2.imshow('orig', img)
            # cv2.waitKey(0)
            
            # Apply some preprocessing
            # images_hist_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            # cv2.imshow('images_hist_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_hist_path'], 'test', 'train') + "/" + filename, img)

            # images_gauss_path
            img = cv2.imread(im_name)
            img = preprocess.gauss_img_sharpening(img)
            # cv2.imshow('images_gauss_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_gauss_path'], 'test', 'train') + "/" + filename, img)

            # images_hist_gauss_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.gauss_img_sharpening(img)
            # cv2.imshow('images_hist_gauss_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_hist_gauss_path'], 'test', 'train') + "/" + filename, img)

            # images_invert_path
            img = cv2.imread(im_name)
            img = preprocess.invert(img)
            # cv2.imshow('images_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_invert_path'], 'test', 'train') + "/" + filename, img)

            # images_hist_invert_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_hist_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_hist_invert_path'], 'test', 'train') + "/" + filename, img)

            # images_gauss_invert_path
            img = cv2.imread(im_name)
            img = preprocess.gauss_img_sharpening(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_gauss_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_gauss_invert_path'], 'test', 'train') + "/" + filename, img)

            # images_hist_gauss_invert_path
            img = cv2.imread(im_name)
            img = preprocess.histogram_equlization_rgb(img)
            img = preprocess.gauss_img_sharpening(img)
            img = preprocess.invert(img)
            # cv2.imshow('images_hist_gauss_invert_path', img)
            # cv2.waitKey(0)
            cv2.imwrite(str.replace(self.config['images_hist_gauss_invert_path'], 'test', 'train') + "/" + filename, img)

    def run_evaluation(self):
        # im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        im_list = sorted(glob.glob(self.config['images_hist_gauss_path'] + '/*.png', recursive=True))
        iou_arr = []
        preprocess = Preprocess()
        eval = Evaluation()
        
        # Change the following detector and/or add your detectors below
        # import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        # cascade_detector = cascade_detector.Detector()

        from yolov5 import detect
        

        for im_name in im_list:
            
            # Read an image
            img = cv2.imread(im_name)

            # Apply some preprocessing
            # img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse
            
            # Run the detector. It runs a list of all the detected bounding-boxes. In segmentor you only get a mask matrices, but use the iou_compute in the same way.
            # prediction_list = cascade_detector.detect(img)
            
            prediction_list = detect.run(weights='results/hist_gauss_4_best.pt', source=im_name, nosave=True, save_txt=True)

            # Read annotations:
            annot_name = os.path.join(self.annotations_path, Path(os.path.basename(im_name)).stem) + '.txt'
            annot_list = self.get_annotations(annot_name)
            prediction_last = glob.glob('runs/detect/*', recursive=True)
            prediction_last.sort(key=os.path.getmtime)
            prediction_name = os.path.join(prediction_last.pop(), 'labels', Path(os.path.basename(im_name)).stem) + '.txt'
            prediction_list = []
            if (os.path.exists(prediction_name)):
                prediction_list = self.get_annotations(prediction_name)

            # print(prediction_list, annot_list)
            x = len(img[0])
            y = len(img)
            prediction_list_2 = []
            for item in prediction_list:
                prediction_list_2.append([int(item[0] * x), int(item[1] * y), int(item[2] * x), int(item[3] * y)])
            annot_list_2 = []
            for item in annot_list:
                annot_list_2.append([int(item[0] * x), int(item[1] * y), int(item[2] * x), int(item[3] * y)])
            # Only for detection:
            p, gt = eval.prepare_for_detection(prediction_list_2, annot_list_2)
            
            iou = eval.iou_compute(p, gt)
            iou_arr.append(iou)

        miou = np.average(iou_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
    # ev.run_preprocessing()