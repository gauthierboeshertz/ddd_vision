from segment_garment import process_seg
import argparse
import os 
import pickle
import json


def count_garment_cat(img_list_path,annos_dir):

    with open(img_list_path, "rb") as fp:
        img_list = pickle.load(fp)

    print('Total images', len(list(set(img_list))))
    count_dict = {}
    for img_num in img_list:
        with open(os.path.join(annos_dir,img_num+'.json')) as json_file:
            img_anno = json.load(json_file)

        for item in img_anno.keys():
            if 'item' not in item:
                continue
            item_name = img_anno[item]['category_name'] 
            if  item_name not in count_dict.keys():
                 count_dict[item_name] = 1
            else:
                count_dict[item_name] = count_dict[item_name] + 1

    total_garment = 0
    for k in count_dict.keys():
        total_garment += count_dict[k]
        print('For ',k, count_dict[k])

    print('Total',total_garment)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_list', type=str)
    parser.add_argument('--annos_dir', type=str)

    args = parser.parse_args()

    count_garment_cat(args.img_list, args.annos_dir)
