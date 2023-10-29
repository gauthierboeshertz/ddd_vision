import os
import pickle 
mesh_folder = '/cluster/scratch/gboeshertz/data/meshes/'
already_done = '/cluster/scratch/gboeshertz/data/united_img_number_done'

with open(already_done, "rb") as fp:
    already_done = pickle.load(fp)

new_meshs = 0
# fetch all files
all_meshes = os.listdir(mesh_folder)
print('N meshes',len(all_meshes))
for file_name in all_meshes:
    # construct full file path
    if '.obj' not in file_name:
        continue
    if file_name.split('_')[0] in already_done:
        os.remove(os.path.join(mesh_folder,file_name))
print('NUmber of new meshes:',new_meshs)
