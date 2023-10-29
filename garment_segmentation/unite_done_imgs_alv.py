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


    all_done = []
    for i in range(400,600):
        done_div_name = os.path.join(args.folder_path, 'img_divs/img_number_done_div' + str(i))
        with open(done_div_name, "rb") as fp:
            b = pickle.load(fp)
        all_done.extend(b)

    print('Have done',len(all_done),' images')
    united_done_div_name = os.path.join(args.folder_path, 'united_img_number_done')
    with open(united_done_div_name, "wb") as fp:
        pickle.dump(all_done, fp)


