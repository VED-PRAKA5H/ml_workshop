import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser(description='Colorize Gray')
parser.add_argument('--input', help='please give path to image')
args = parser.parse_args()

if args.input is None:
    print('please give the imput greyscale image.')
    print('Usage example: python fill_color.py --input greyscaleimage.png')
    exit()

if os.path.isfile(args.input) == 0:
    print("Input file does not exist")
    exit()

frame = cv2.imread(args.input)
protofile = "../model/"
weightsfile = "../model/"
pts_in_hull = np.load("../model/")

net = cv2.dnn.readNetFromCaffe(protofile, weightsfile)

pts_in_hull = pts_in_hull.trasnpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.flo)]

