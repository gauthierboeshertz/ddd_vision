import argparse
import torch
import os
from os.path import isfile, join,isdir
import shutil
import copy
import numpy as np
import json


# only save pictures with one person
# discard images with more than half keypoints are bad  confidence < threshold
# or images where not enough keypoints

def process_train_folder_annos(img_path, annos_path, out_path):
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = os.listdir(img_path)
    images = [img for img in images if os.path.isfile(os.path.join(img_path,img))]
    num_images_copied = 0
    num_images_user = 0
    num_images_only_clothes = 0
    for img in images:
        with open(os.path.join(annos_path,img.split('.')[0]+'.json')) as json_file:
            img_annos = json.load(json_file)
        
        if img_annos['source'] == 'user' :
            num_images_user += 1 
            continue
        if 'item1' in img_annos.keys():
            if img_annos['item1']['viewpoint'] == 1:
                num_images_only_clothes += 1
                continue
        shutil.move(os.path.join(img_path,img), os.path.join(out_path,img))
        num_images_copied += 1
    print('Found ',num_images_user, ' that were marked as user')
    print('Found ',num_images_only_clothes, ' with no one in it')
    print('Found ',num_images_copied,' good images out of the',len(images),' total images')

def process_train_folder_seg(img_path,kps_path, seg_path, out_path, threshold_kps = 5,use_seg=False):
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = os.listdir(img_path)
    images = [img for img in images if os.path.isfile(os.path.join(img_path,img))]
    num_images_copied = 0
    for img in images:
        with open(os.path.join(kps_path,img.split('.')[0]+'.json')) as json_file:
            img_kps = json.load(json_file)
        if len(img_kps.keys()) != 1:
            continue
        kps = img_kps['0']['kps']
        
        try:
            seg = np.load(os.path.join(seg_path,img.split('.')[0]+'.npy'))
        except:
            print('Seg for ',img,' didnt exist')
            continue 
        num_kp_not_in_seg = 0
        for kp_idx in range(len(kps)):
            if kps[kp_idx][2] > 0.5:
                square_relax = 5 
                if seg[ int(kps[kp_idx][1])-square_relax:int(kps[kp_idx][1])+square_relax,
                        int(kps[kp_idx][0])-square_relax:int(kps[kp_idx][0])+square_relax,0].sum() < 1:
                    num_kp_not_in_seg += 1
        if num_kp_not_in_seg > 3:
            continue
        # if all good then copy to new folder
        shutil.move(os.path.join(img_path,img), os.path.join(out_path,img))
        num_images_copied += 1
    print('Found ',num_images_copied,' good images out of the',len(images),' total images')

def process_train_folder_kps(img_path, out_path, kps_path, threshold_kps = 5):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images = os.listdir(img_path)
    images = [img for img in images if os.path.isfile(os.path.join(img_path,img))]
    num_images_copied = 0
    for img in images:
        with open(os.path.join(kps_path,img.split('.')[0]+'.json')) as json_file:
            img_kps = json.load(json_file)
        if len(img_kps.keys()) != 1:
            continue
        num_bad_kps = 0
        kps = img_kps['0']['kps']
        kps_subset = img_kps['0']['subset']
        # kp correspondance on pytorch-openpose git
        dc_kps = [8,9,10,11,12,13]  
        head_kps = [14,15,16,17]
        num_heads = 0
        for kp_idx in range(len(kps)):
            if kps[kp_idx][3] in head_kps:
                num_heads += 1
            if kps[kp_idx][2] < 0.5 and not (kps[kp_idx][3] in dc_kps ):
                num_bad_kps += 1
        if num_bad_kps > threshold_kps:
            continue
        if num_heads <=1 :
            continue 
        # if all good then copy to new folder
        shutil.copyfile(os.path.join(img_path,img), os.path.join(out_path,img))
        num_images_copied += 1
    print('Found ',num_images_copied,' good images out of the',len(images),' total images')

def process_val_folder(data_path, threshold_kps = 8):
    
    images_folder = data_path+'/images/'
    images_for_icon = data_path + '/images_for_icon/'
    if not os.path.exists(images_for_icon):
        os.makedirs(images_for_icon)

    kps_folder = data_path+ '/images_pose/'
    images = os.listdir(images_folder)

    num_images_copied = 0
    for img in images:
        with open(kps_folder+img.split('.')[0]+'_keypoints.json') as json_file:
            img_kps = json.load(json_file)
        if not (len(img_kps['people']) == 1):
            continue
        kps = img_kps['people'][0]['pose_keypoints_2d']
        #https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html#autotoc_md40
        #There are 17 kps per person keep only those with more than 8 in the upper body
        # lower body kps = 10, "RKnee, 11, "RAnkle,  13, "LKnee, 14, "LAnkle, 19, "LBigToe, 20, "LSmallToe", 21, "LHeel, 22, "RBigToe,23, "RSmallToe,24, "RHeel, 25 background
        num_bad_kps = 0
        dc_kps = [10,11,13,14,19,20,21,22,23,24,25]
        for kp_idx in range(2,len(kps),3):
            if kps[kp_idx] < 0.1 and not (kp_idx in dc_kps ):
                num_bad_kps += 1
        if num_bad_kps > threshold_kps:
            continue
        # if all good then copy to new folder
        shutil.copyfile(images_folder+img, images_for_icon+img)
        num_images_copied += 1
    print('Found ',num_images_copied,' good images out of the',len(images),' total images')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--img_path', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--kps_path', type=str)
    parser.add_argument('--seg_path', type=str)
    parser.add_argument('--annos_path', type=str)

    parser.add_argument('--threshold_kps', type=int, default=8)
    parser.add_argument('--use_seg', action='store_true')
    parser.add_argument('--use_kps', action='store_true')
    parser.add_argument('--use_annos', action='store_true')

    args = parser.parse_args()
    if 'train' in args.img_path:
        if args.use_kps:
            process_train_folder_kps(args.img_path,out_path=args.out_path,kps_path=args.kps_path,
                threshold_kps=args.threshold_kps)
        if args.use_seg:
            process_train_folder_seg(args.img_path,seg_path=args.seg_path, kps_path=args.kps_path,out_path=args.out_path,
                threshold_kps=args.threshold_kps)
        if args.use_annos:
            process_train_folder_annos(args.img_path, annos_path=args.annos_path, out_path=args.out_path)

    else:
        process_val_folder(args.data_path,threshold_kps=args.threshold_kps)
