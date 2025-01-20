import os
import numpy as np
import cv2
from pathlib import Path
#from mean_std import get_mean, get_std

feedback_samples = 4

def get_mean(path):
    imageFilesDir = Path(r'' + path)
    files = list(imageFilesDir.rglob('*.jpg'))
    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    numSamples = len(files)
    feedback = numSamples // feedback_samples
    print("Calculating the std of {} samples and you will get a feedback each {}".format(numSamples, feedback))
    for i in range(numSamples):
        if i % feedback == feedback - 1:
            print("Feedback point: ", i + 1)
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            mean[j] += np.mean(im[:, :, j])
    mean = (mean / numSamples)
    return mean


def get_std(path):
    imageFilesDir = Path(r'' + path)
    files = list(imageFilesDir.rglob('*.jpg'))
    mean = np.array([0., 0., 0.])
    stdTemp = np.array([0., 0., 0.])
    std = np.array([0., 0., 0.])
    numSamples = len(files)
    feedback = numSamples // feedback_samples
    print("Calculating the std of {} samples and you will get a feedback each {}".format(numSamples, feedback))
    for i in range(numSamples):
        if i % feedback == feedback - 1:
            print("Feedback point: ", i + 1)
        im = cv2.imread(str(files[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])
    std = np.sqrt(stdTemp / numSamples)
    return std

