"""
Reconstruct shapes using a trained model.
"""

import sys
import os
sys.path.append('..')
sys.path.append('../..')

####################################################################################
####################################################################################
#               HACK FOR RUNNING INSIDE VSCODE
#sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
thispath =  os.path.dirname( os.path.abspath(__file__) )
os.chdir(thispath)
sys.path.append('/home/kristofe/Documents/Projects/metasdf/3D')
####################################################################################
####################################################################################
import torch
import curriculums
from reconstruction import reconstruct
import levelset_data
import json
import argparse

device = torch.device('cuda')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--checkpoint', default='latest.pth')
    args = parser.parse_args()
    
    args.exp_name = 'planes_test_pe'
    curriculum = getattr(curriculums, args.exp_name)
        
    
    checkpoint = torch.load(curriculum['output_dir'] + '/' + args.checkpoint, map_location=device)
    model = (checkpoint['model'].module)
    reconstruction_output_dir = curriculum['reconstruction_output_dir']

    with open(curriculum['test_split'], "r") as f:
        split = json.load(f)
    
    data_source = curriculum['data_source']
    npz_filenames = levelset_data.get_instance_filenames(data_source, split)

    reconstruct(model, npz_filenames, reconstruction_output_dir, data_source=data_source)