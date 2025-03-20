# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from .depth_estimation import networks
from .depth_estimation.layers import disp_to_depth
from .depth_estimation.utils import download_model_if_doesnt_exist
from .depth_estimation.evaluate_depth import STEREO_SCALE_FACTOR

def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val) if max_val > min_val else torch.zeros_like(tensor)


def create_depth_estimator():
    model = DepthEstimator()
    if torch.cuda.is_available():
        model.device = torch.device("cuda")
    else:
        model.device = torch.device("cpu")
    
    download_model_if_doesnt_exist("mono+stereo_640x192")
    model_path = os.path.join("models", "mono+stereo_640x192")
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=model.device)

    # extract the height and width of image that this model was trained with
    model.feed_height = loaded_dict_enc['height']
    model.feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(model.device)
    encoder.eval()
    model.encoder = encoder

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=model.device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(model.device)
    depth_decoder.eval()
    model.depth_decoder = depth_decoder

    return model

class DepthEstimator():
    def __init__(self):
        self.feed_height = None
        self.feed_width = None
        self.encoder = None
        self.depth_decoder = None
        self.device = None


    def forward(self, img):

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():

            # Load image and preprocess
            # Convert to PIL image
            to_pil = transforms.ToPILImage()
            input_image = to_pil(img)
            input_image.convert('RGB')
            #display_img = input_image
            original_width, original_height = input_image.size
            input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Returning numpy vector
            #scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            #metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

            # Normalize
            disp_resized_norm = normalize_tensor(disp_resized)

            # Display colormapped depth image
            # disp_resized_np = disp_resized.squeeze().cpu().numpy()
            # vmax = np.percentile(disp_resized_np, 95)
            # normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            # im = pil.fromarray(colormapped_im)
            # display_img.show()
            # im.show()

            return disp_resized_norm
