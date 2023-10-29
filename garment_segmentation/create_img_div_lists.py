from segment_garment import process_seg
import argparse
import os 
import pickle


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--number_div', type=int)

    args = parser.parse_args()
    images_path = os.path.join(args.folder_path,'images')

    imgs = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    imgs_divs = []
    for i in range(args.number_div):
        div_start = i*(len(imgs)//args.number_div)
        div_end = (i+1)*(len(imgs)//args.number_div)-1
        print(div_start,div_end)
        imgs_divs.append([imgs[img_index].split('.')[0] for img_index in range(div_start,div_end)])

        print('len of div i',len(imgs_divs[i]))

    img_divs_folder = os.path.join(args.folder_path,'img_divs')
    if not os.path.isdir(img_divs_folder):
        os.mkdir(img_divs_folder)

    for i in range(args.number_div):
        img_div_name = os.path.join(img_divs_folder,'img_number_to_do_div' + str(i)) 
        if not os.path.isfile(img_div_name):
            with open(img_div_name, "wb") as fp:   
                pickle.dump(imgs_divs[i], fp)

    for i in range(args.number_div):
        done_div_name = os.path.join(img_divs_folder,'img_number_done_div' + str(i)) 
        if not os.path.isfile(done_div_name):
            with open(done_div_name, "wb") as fp:   
                pickle.dump([], fp)

    united_done_imgs_name = os.path.join(args.folder_path,'united_img_number_done')
    if not os.path.isfile(united_done_imgs_name):
        with open(united_done_imgs_name, "wb") as fp:   
            pickle.dump([], fp)
