import time 
import numpy as np
import scipy

def main():
    return

def load_data(path_data="../data/"):
    ori_data = scipy.io.loadmat(path_data+"ori.mat")
    ori_noise_10 = scipy.io.loadmat(path_data+"ori_noise_10.mat")
    return ori_data, ori_noise_10 

if __name__ == "__main__" :
    main()