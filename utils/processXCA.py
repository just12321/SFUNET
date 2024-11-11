import os
import h5py
import PIL.Image as Image
import numpy as np

def processXCA(xcaFile, outDir, ratio=0.8, type='image'):
    with h5py.File(xcaFile, 'r') as f:
        # Get the number of frames
        numFrames = len(f['image'])
        # Loop through each frame and save the image to a separate file
        for i in range(numFrames):
            # Get the image data for the current frame
            image = f['image'][i][0].astype(np.uint8)
            if image.ndim == 3: image=image[0]

            # Save the image to a file
            outFileName = os.path.join(outDir, f'{"train" if (i < numFrames * ratio) else "val"}/{type}/{i}.png')
            Image.fromarray(image).save(outFileName)

if __name__ == '__main__':
    processXCA(r'F:\开题报告\Codes\Dataset\XCA/test1_ori.hdf5', r'F:\开题报告\Codes\Project\dataset\DatasetXCA', type='image')
    processXCA(r'F:\开题报告\Codes\Dataset\XCA/test1_gt.hdf5', r'F:\开题报告\Codes\Project\dataset\DatasetXCA', type='manual')
    print('Done!')

