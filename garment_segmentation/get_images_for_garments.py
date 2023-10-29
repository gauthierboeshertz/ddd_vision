from segment_garment import process_seg
import argparse
import os 
import pickle
import json

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str)

    args = parser.parse_args()
    annos_folder = os.path.join(args.folder_path,'annos')
    all_annos = os.listdir(annos_folder)
    all_annos = [a for a in all_annos if '.json' in a]

    with open(os.path.join(args.folder_path,'united_img_number_done'), "rb") as fp:
        done_imgs = pickle.load(fp)
    garment_to_imgs = {}
    for anno_path in all_annos:

        if anno_path.split('.')[0] not in done_imgs:
            continue

        with open(os.path.join(annos_folder,anno_path)) as json_file:
            anno = json.load(json_file)

        for item in anno.keys():
            if 'item' not in item:
                continue
            item_name = anno[item]['category_name'] 
            if  item_name not in garment_to_imgs.keys():
                garment_to_imgs[item_name] = [anno_path.split('.')[0]]
            else:
                garment_to_imgs[item_name].append(anno_path.split('.')[0])
    for k in garment_to_imgs.keys():
        print('--'*20)
        print('For garment',k)
        print(garment_to_imgs[k][:50])