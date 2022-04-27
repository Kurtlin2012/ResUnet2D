import os
import argparse
import sys
import numpy as np
import pydicom
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2


def parse_args(args):
    parser = argparse.ArgumentParser(description='Unet2D predict')
    parser.add_argument('--weight', type = str, help = 'File path to load the weights(h5 file).')
    parser.add_argument('--source', type = str, help = 'Folder path of the testing data(dicom file).')
    parser.add_argument('--image-size', type = int, default = 512, help = 'Image sizes, default is 512.')
    parser.add_argument('--output', type = str, help = 'Folder path to save the generated reports.')
    parser.add_argument('--channel', type = int, default = 0, help = 'Index of channel which needs to export. Default is 0.')
    return parser.parse_args(args)

def main(args = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    # load the model
    model = load_model(args.weight, compile=False)
    
    # load the 3d image matrix
    file_list = os.listdir(args.source)
    for file in file_list:
        dcm = pydicom.dcmread(f'{args.source}/{file}')
        X = dcm.pixel_array
        pixel = dcm.PixelSpacing[0] * dcm.PixelSpacing[1] * dcm.SliceThickness
        X = ((X - X.min()) / (X.max() - X.min() + 1e-6)).astype(np.float32)
        X = np.expand_dims(np.expand_dims(X, axis=0), axis=3)

        # predict one case each time
        pred = model.predict(X)
        y = np.round(pred[0, ..., args.channel])
        area = np.sum(y, keepdims=False) * pixel
        
        # Plot the image
        img = cv2.cvtColor(X[0,...,0], cv2.COLOR_GRAY2BGR)
        pred = img.copy()
        pred[y, 2] = 1
        plt.figure(); 
        plt.suptitle(f'Area of WMH: {area:.2f} mm^3', fontsize=20);
        plt.subplot(121); plt.imshow(img); plt.axis('off'); plt.title('Original');
        plt.subplot(122); plt.imshow(pred); plt.axis('off'); plt.title('Predict');
        plt.savefig(f'{args.output}/{file[:-4]}.png', dpi=350, bbox_inches='tight');
        plt.close();
        
        np.save(f'{args.output}/{file[:-4]}.npy', pred)
        

if __name__ == '__main__':
    main()