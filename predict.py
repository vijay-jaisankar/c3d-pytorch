""" How to use C3D network. """
import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import cv2

"""
    Extracts all frames from the specified filename
"""
def get_all_frames(filename):

    # Setting up VideoCapture
    cap = cv2.VideoCapture(filename)

    list_frames = []
    # Looping throught the frames
    for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        # Read frame 
        ret, frame = cap.read()
        
        # Store the frame 
        list_frames.append(frame)

    # Close down everything
    cap.release()
    return list_frames



def get_sport_clip(clip_name, clip_provided = None, verbose=False):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    if clip_provided is None:
        clip = sorted(glob(join('data', clip_name, '*.png')))
        clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in clip])
        clip = clip[:, :, 44:44+112, :]  # crop centrally
    
    else:
        clip = clip_provided
        clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in clip])
        clip = clip[:, :, 44:44+112, :]  # crop centrally 

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)
    print(f"Input dimensions: {clip.shape}")

    return torch.from_numpy(clip)


def main():
    """
    Main function.
    """

    frames = get_all_frames("/home/vijay/Desktop/labs/ibm/c3d-pytorch/ssbd/headbanging_sample.mp4")

    # load a clip to be predicted
    X = get_sport_clip('roger', frames)
    X = Variable(X)
    X = X.cuda()

    # get network pretrained model
    net = C3D()
    net.load_state_dict(torch.load('c3d_weights.pickle'))
    net.cuda()
    net.eval()

    # perform prediction
    prediction = net(X)
    prediction = prediction.data.cpu().numpy()
    predictions_proc = prediction[0].reshape(-1,1)
    print(predictions_proc[0].shape, predictions_proc[:10])



# entry point
if __name__ == '__main__':
    main()
