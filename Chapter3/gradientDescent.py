#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
    help="size of SGD mini-batches")
ap.add_argument("-ep", "--epsilon", type=float, default=0.01
    help="epsilon value")
ap.add_argument("-g", "--gamma", type=float, default=0.9,
    help="gamma value")

args = vars(ap.parse_args())

