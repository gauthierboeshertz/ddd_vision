from segment_garment import process_seg
import argparse
import os 
import pickle


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--number_div_done', type=int)
    parser.add_argument('--number_div', type=int)

    args = parser.parse_args()
    images_path = os.path.join(args.folder_path,'images')
    meshes_path = os.path.join(args.folder_path,'meshes')
    img_divs_path = os.path.join(args.folder_path,'img_divs')

    imgs_already_div = []
    for i in range(args.number_div_done):
        done_div_name = os.path.join(img_divs_path,'img_number_to_do_div' + str(i))
        with open(done_div_name, "rb") as fp:
            b = pickle.load(fp)
        imgs_already_div.extend(b)

    print('Found ',len(imgs_already_div),' images already divided')
    imgs_to_div =  [f.split('.')[0] for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path,f))]
    imgs_to_div = [img for img in imgs_to_div if img not in imgs_already_div]
    print('Found ',len(imgs_to_div),' NEW images to div')

    imgs_divs = []

    for i in range(args.number_div):
        div_start = i*(len(imgs_to_div)//args.number_div)
        div_end = (i+1)*(len(imgs_to_div)//args.number_div)-1
        print(div_start,div_end)
        imgs_divs.append([imgs_to_div[img_index] for img_index in range(div_start,div_end)])

        print('len of div i',len(imgs_divs[i]))

    img_divs_folder = os.path.join(args.folder_path,'img_divs')

    for i in range(args.number_div):
        img_div_name = os.path.join(img_divs_folder,'img_number_to_do_div' + str(i+args.number_div_done)) 
        with open(img_div_name, "wb") as fp:   
            pickle.dump(imgs_divs[i], fp)

    for i in range(args.number_div):
        done_div_name = os.path.join(img_divs_folder,'img_number_done_div' + str(i+args.number_div_done)) 
        if not os.path.isfile(done_div_name):
            with open(done_div_name, "wb") as fp:   
                pickle.dump([], fp)

