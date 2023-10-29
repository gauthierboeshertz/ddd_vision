from tabnanny import verbose
import trimesh
import os, os.path, shutil
import argparse
import json 
import numpy as np
from shapely import geometry as geo
from shapely.geometry import Polygon,LinearRing
import numpy as np
import json
import trimesh
from seg_utils import   query_color_no_visibility, query_color_vis
import cv2
import torch 
from imutils import process_seg
from scipy.spatial import cKDTree
import pymeshfix
import pickle
import tqdm
import sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def widen_shape(shape, pix):
    poly = geo.Polygon(shape) 
    try:
        t = Polygon(poly.buffer(pix).exterior, [LinearRing(shape)])
        newx,newy = t.exterior.coords.xy
        new = np.vstack((newx,newy))
        shape_expanded = new.T
        shape_expanded = shape_expanded.astype(np.int32)
    except :
        shape_expanded = shape
    return shape_expanded

def color_vertices(img_number,  clothes_mask, mesh):
    device = 'cpu'
    body_mask = np.load(os.path.join(segs_path,img_number+'.npy'))

    mask = process_seg(os.path.join(images_path,img_number+'.jpg'),body_mask,clothes_mask).float()

    mask = mask.permute(0,3,1,2)

    vertex_colors = query_color_no_visibility(torch.from_numpy(mesh.vertices.view(np.ndarray)).to(device),
                        torch.from_numpy(mesh.faces.view(np.ndarray)).to(device),
                        (mask).to(device),
                        device=device)[0]
    vertex_colors = vertex_colors.permute(2,1,0)[0]
    return  vertex_colors.cpu().numpy()


def color_vertices_visibility(img_number,  clothes_mask, mesh):
    device = 'cuda'
    body_mask = np.load(os.path.join(segs_path,img_number+'.npy'))

    mask = process_seg(os.path.join(images_path,img_number+'.jpg'),body_mask,clothes_mask).float()


    mask = mask.permute(0,3,1,2)
    vertex_colors = query_color_vis(torch.from_numpy(mesh.vertices.view(np.ndarray)).to(device),
                        torch.from_numpy(mesh.faces.view(np.ndarray)).to(device),
                        (mask).to(device),
                        device=device)[0]
    vertex_colors = vertex_colors.permute(2,1,0)[0]
    return  vertex_colors.cpu().numpy()


def mask_mesh_vertices(mesh,mesh_mask):
    
    mesh = mesh.copy()
    mesh_mask = mesh_mask.copy()

    mesh_mask = np.argwhere(mesh_mask).reshape((-1,))
    good_faces = []
    for i in range(3):
        good_faces.extend(np.where(np.isin(mesh.faces[:,i], mesh_mask))[0].tolist())

    good_faces = np.unique(np.array(good_faces))

    new_mesh_faces = np.zeros(mesh.faces.shape[0],)

    if good_faces.size > 0:
        new_mesh_faces[good_faces] = 1
        new_mesh_faces = new_mesh_faces.astype(np.bool)
        mesh.update_faces(new_mesh_faces)
        mesh.remove_unreferenced_vertices()
    return mesh



def get_shape_mask(img, shape,expansion=0):

    shape_mask = img.copy()

    for poly in shape:
        poly = np.array(poly, dtype=np.int32)
        poly = poly.reshape(poly.shape[0]//2,2)
        poly_expanded = widen_shape(poly,expansion)
        shape_mask = cv2.fillPoly(shape_mask, [poly_expanded], (255,0,0))
    return shape_mask

def whiten_mesh(mesh):
    back_mask = mesh.visual.vertex_colors.view(np.ndarray)[:,:3]
    back_mask = (back_mask ==  [0,0,0]).all(axis=1)
    mesh.visual.vertex_colors[back_mask] = [255,255,255,255]
    return mesh

def smpl_bp_label(smpl,refine,labels,bp_to_idx):
    bp_label = np.zeros((smpl.vertices.shape[0],))

    for bp in bp_to_idx:
        v_list = bp_to_idx[bp]
        bp_label[v_list] = labels.index(bp)

    tree = cKDTree(smpl.vertices.view(np.ndarray))

    refine_to_label = np.zeros((refine.vertices.shape[0],))

    for v_idx, v in enumerate(refine.vertices):
        smpl_vert, smpl_idx = tree.query(v,k=1)
        refine_to_label[v_idx] = bp_label[smpl_idx]
    
    return refine_to_label




def remove_bodyparts(smpl,clothed_smpl,cloth_type):
    
    bp_to_idx = json.load(open('data/smpl_vert_segmentation.json'))

    labels = list(bp_to_idx.keys())

    refine_labels = smpl_bp_label( smpl,clothed_smpl,labels,bp_to_idx)
    bad_bp = ['leftHand', 'rightHand','leftToeBase','leftFoot','rightFoot','head','leftHandIndex1','rightHandIndex1','neck','rightToeBase','leftHand']
    

    short_sleeves_top = [1,3]
    long_sleeve_top = [2,4]
    no_sleeve_top = [5,6]
    bottoms = [7,8,9]
    no_sleeve_dress = [12,13]

    if cloth_type in no_sleeve_top:
        bad_bp += ['leftArm', 'rightArm','leftForeArm', 'rightForeArm','leftLeg','rightLeg']

    if cloth_type in short_sleeves_top:
        bad_bp += ['leftForeArm', 'rightForeArm','leftLeg','rightLeg']

    if cloth_type in long_sleeve_top:
        bad_bp += ['leftLeg','rightLeg']

    if cloth_type in bottoms : 
        bad_bp += ['leftArm', 'rightArm','leftForeArm', 'rightForeArm']

    if cloth_type == 10: #short sleeve dress
        bad_bp += ['leftForeArm', 'rightForeArm']

    if cloth_type in no_sleeve_dress: #short sleeve dress
        bad_bp += ['leftForeArm', 'rightForeArm','leftArm', 'rightArm']


    verts_mask = np.ones((clothed_smpl.vertices.shape[0]))

    for bp in bad_bp:
        bp_label = labels.index(bp)
        verts_mask[refine_labels == bp_label] = 0

    return verts_mask


def remove_filled_bodyparts(smpl,clothed_smpl,cloth_type):
    
    bp_to_idx = json.load(open('data/smpl_vert_segmentation.json'))

    labels = list(bp_to_idx.keys())

    refine_labels = smpl_bp_label( smpl,clothed_smpl,labels,bp_to_idx)
    bad_bp = []
    
    short_sleeves_top = [1,3]
    long_sleeve_top = [2,4]
    no_sleeve_top = [5,6]
    bottoms = [7,8,9]
    no_sleeve_dress = [12,13]

    if cloth_type in no_sleeve_top:
        bad_bp += ['leftArm', 'rightArm','leftForeArm', 'rightForeArm','leftShoulder', 'rightShoulder']

    if cloth_type in short_sleeves_top:
        bad_bp += ['leftForeArm', 'rightForeArm','leftArm', 'rightArm']

    if cloth_type in long_sleeve_top:
        bad_bp += ['leftForeArm', 'rightForeArm']

    if cloth_type in [8]: 
        bad_bp += ['rightUpLeg', 'leftUpLeg','leftLeg','rightLeg']

    if cloth_type in [7,9]: 
        bad_bp += ['leftLeg','rightLeg']

    if cloth_type == 10: #short sleeve dress
        bad_bp += ['leftForeArm', 'rightForeArm','leftShoulder', 'rightShoulder']

    if cloth_type in no_sleeve_dress: #short sleeve dress
        bad_bp += ['leftForeArm', 'rightForeArm','leftShoulder', 'rightShoulder']


    verts_mask = np.ones((clothed_smpl.vertices.shape[0]))

    for bp in bad_bp:
        bp_label = labels.index(bp)
        verts_mask[refine_labels == bp_label] = 0

    return verts_mask


def keep_bodyparts(smpl,clothed_smpl,cloth_type):
    
    bp_to_idx = json.load(open('data/smpl_vert_segmentation.json'))

    labels = list(bp_to_idx.keys())

    refine_labels = smpl_bp_label( smpl,clothed_smpl,labels,bp_to_idx)
    
    short_sleeves_top = [1]
    long_sleeve_top = [2]
    no_sleeve_top = [5,6]
    no_sleeve_dress = [12,13]

    good_bp = []
    if cloth_type in no_sleeve_top:
        good_bp = ['spine1', 'spine2','spine']

    if cloth_type == 3:
        good_bp = []

    if cloth_type in short_sleeves_top:
        good_bp = ['spine1', 'spine2','leftShoulder', 'rightShoulder','spine']

    if cloth_type == 4:
        good_bp = ['leftArm','rightArm','leftForeArm','rightForeArm']

    if cloth_type in long_sleeve_top:
        good_bp = ['spine1', 'spine2','leftShoulder', 'rightShoulder','spine','leftArm','rightArm']

    #if cloth_type  == 7 : #or cloth_type  == 9 : # only short  
    #    good_bp = ['rightUpLeg', 'leftUpLeg']

    if cloth_type  == 8 : #pants
        good_bp = ['rightUpLeg', 'leftUpLeg','leftLeg','rightLeg' ]

    if cloth_type == 10: #short sleeve dress
        good_bp = ['spine1', 'spine2','leftShoulder', 'rightShoulder','spine'] 

    if cloth_type == 11: #long sleeve dress
        good_bp = ['spine1', 'spine2','leftShoulder', 'rightShoulder','spine','leftArm','rightArm',] 

    if cloth_type in no_sleeve_dress: #short sleeve dress
        good_bp = ['spine1', 'spine2','spine']

    verts_mask = np.zeros((clothed_smpl.vertices.shape[0]))

    for bp in good_bp:
        bp_label = labels.index(bp)
        verts_mask[refine_labels == bp_label] = 1

    return verts_mask

def mesh_bounding_box(mesh):
    vertices = mesh.vertices
    max_x = np.max(vertices[:,0])
    min_x = np.min(vertices[:,0])
    max_y = np.max(vertices[:,1])
    min_y = np.min(vertices[:,1])
    max_z= np.max(vertices[:,2])
    min_z= np.min(vertices[:,2])

    return [min_x,max_x,min_y,max_y,min_z,max_z]


def mask_outside_bb(mesh,bb):
    vertices = mesh.vertices
    v_mask = (vertices[:,0] >= bb[0]) &  (vertices[:,0] <= bb[1])\
          &  (vertices[:,1] >= bb[2]) &  (vertices[:,1] <= bb[3]) \
         &   (vertices[:,2] >= bb[4]) &  (vertices[:,2] <= bb[5])
    return v_mask

def get_skinmask(img):
    frame = img
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    return skinMask


def get_skin_vertices(img_number,img, mesh):

    mesh = mesh.copy()
    skinmask = get_skinmask(img)
    skinmask = np.expand_dims(skinmask,-1)
    skinmask = np.repeat(skinmask,3,axis=-1)
    vertex_mask = color_vertices_visibility(img_number,  skinmask, mesh)
    skin_vertex  = (vertex_mask[:,0] > 240)
    vertex_to_remove = mesh.vertices[skin_vertex]
    skin_vertex =  ~skin_vertex
    mesh_firs_layer_removed = mask_mesh_vertices(mesh,skin_vertex)

    vertex_mask = color_vertices_visibility(img_number,  skinmask, mesh_firs_layer_removed)
    skin_vertex  = (vertex_mask[:,0] > 240)
    vertex_to_remove = np.vstack((vertex_to_remove,mesh_firs_layer_removed.vertices[skin_vertex]))
    return vertex_to_remove
    #downpandedclothed_body_colors_v_col = clothed_body.visual.vertex_colors[downpandedclothed_body_colors_v][:,:3]

def keep_biggest_comp(mesh):
    split = mesh.split(only_watertight=False, engine='networkx')

    biggest_mesh = None
    for comp in split:
        if biggest_mesh is None or comp.vertices.shape[0] > biggest_mesh.vertices.shape[0]:
            biggest_mesh = comp
    return biggest_mesh

def fill_holes(mesh,num_edges=50):
    mesh = mesh.copy()

    tin = pymeshfix.PyTMesh()
    
    with HiddenPrints():
        tin.load_array(mesh.vertices, mesh.faces) 
        tin.fill_small_boundaries(nbe=num_edges, refine=True)
        vert, faces = tin.return_arrays()
    final = trimesh.Trimesh(vert,
                            faces,
                            process=False,
                            maintains_order=True)
    return final 

def remove_filling(filled_mesh,mesh,smpl,cloth_type):

    mesh_bb = mesh_bounding_box(mesh)
    filled_vertices_keep = np.zeros((filled_mesh.vertices.shape[0],))
    old_vertices = mesh.vertices
    top_y = mesh_bb[3]
    bottom_y = mesh_bb[2]
    top_fill_limit  = top_y - (top_y - bottom_y)/5
    bottom_fill_limit  = bottom_y + (top_y - bottom_y)/5


    verts_to_keep_bp = remove_filled_bodyparts(smpl,filled_mesh,cloth_type)
    for i,new_vert in enumerate(filled_mesh.vertices):
        filled_vertices_keep[i] =  (new_vert == old_vertices).all(axis=1).any()
    
    filled_vertices_keep +=  verts_to_keep_bp * ((filled_mesh.vertices[:,1] < top_fill_limit) * (filled_mesh.vertices[:,1] > bottom_fill_limit))
    filled_vertices_keep[filled_vertices_keep >=1] = 1
    
    good_mesh = mask_mesh_vertices(filled_mesh,filled_vertices_keep)
    return good_mesh



def get_middle_z_mask(mesh):
    return mesh[mesh.vertices[:,2] > mesh.vertices[:,2].mean()]

def get_middle_cloth_mask(img, seg):    

    if seg["category_id"] == 4:
        keep_lds = np.array([4,38,39,20,37,35,36,34]) - 1

    if seg["category_id"] == 3:
        keep_lds = np.array([2,30,31,16,29,28,27,26]) - 1
    keep_lds = keep_lds.astype(np.int32)
    lds = np.array(seg['landmarks'])
    lds = lds.reshape((lds.shape[0]//3,3))
    lds = lds[keep_lds]
    lds = lds[lds[:,2]!= 0][:,:2]
    lds = lds.astype(np.int32)
    clothes_mask_t = img.copy()
    clothes_mask_t = cv2.fillPoly(clothes_mask_t, [lds], (255,0,0))

    return clothes_mask_t


def overlap_cloth_body(body_seg,cloth_seg):

    body_seg = body_seg.copy()[:,:,0]
    cloth_seg = cloth_seg.copy()[:,:,0]
    body_seg[body_seg>0] = 1
    cloth_seg[cloth_seg>0] = 1
    return (cloth_seg * body_seg).sum()/cloth_seg.sum()
     

def segment_outwear(item,item_name,mask_zeros,img_number,clothed_body,smpl):
    item_segm = item['segmentation']

    img = cv2.imread(os.path.join(images_path,img_number+'.jpg'))

    # get mask of the img that is between the two cloth part
    # then get the corresponding vertices and get the mask of the shown part
    middle_cloth_mask = get_middle_cloth_mask(np.zeros_like(img), item)
    middle_cloth_mask_colors = color_vertices(img_number,middle_cloth_mask,clothed_body)
    middle_cloth_mask_colors  = (middle_cloth_mask_colors[:,0] > 240)
    middle_vertices_to_remove = middle_cloth_mask_colors * (clothed_body.vertices[:,2] > np.mean(clothed_body.vertices[middle_cloth_mask_colors,2]))



    # get the shape of the cloth segmentation in an image, expand it to capture all fo the cloth
    expanded_clothes_mask = get_shape_mask(mask_zeros, item_segm, 10).astype(np.int32)
    expanded_clothes_mask += middle_cloth_mask.astype(np.int32)
    expanded_clothes_mask[expanded_clothes_mask>0] = 255
    expanded_clothes_mask = expanded_clothes_mask.astype(np.uint8)

    clothed_body_colors_mask = color_vertices(img_number,expanded_clothes_mask,clothed_body)
    clothed_body_colors_mask  = (clothed_body_colors_mask[:,0] > 240)
    clothed_body_colors_v = -np.ones_like(clothed_body.visual.vertex_colors.view(np.ndarray))[:,:3]
    clothed_body_colors_v[clothed_body_colors_mask]  = clothed_body.visual.vertex_colors[clothed_body_colors_mask][:,:3]

    #get the color of the cloth by using a smaller mask
    downpanded_clothes_mask = get_shape_mask(mask_zeros, item_segm, -5)
    downpandedclothed_body_colors = color_vertices(img_number,downpanded_clothes_mask,clothed_body)
    downpandedclothed_body_colors_v  = (downpandedclothed_body_colors[:,0] > 240)
    downpandedclothed_body_colors_v_col = clothed_body.visual.vertex_colors[downpandedclothed_body_colors_v][:,:3]

    cloth_colors = np.unique(downpandedclothed_body_colors_v_col,axis=0)

    # get vertices that have the cloth color
    isin = np.zeros((clothed_body_colors_v.shape[0],),dtype=np.int32)
    for i,col in enumerate(clothed_body_colors_v):
        isin[i] = int((col == cloth_colors).all(axis=1).any())
    
    should_work =  clothed_body_colors_mask #& isin#np.isin(clothed_body_colors_v, cloth_colors).all(axis=1)


    # remove bodyparts that should not be in the final mesh 

    #skin_vertices = get_skin_vertices(folder_path,img_number,img, clothed_body)
    #skin_vertices = skin_vertices.tolist()
    #isskin = np.ones((clothed_body.vertices.shape[0],),dtype=np.int32)
    #vertices_list = clothed_body.vertices.tolist()
    #for i,col in enumerate(skin_vertices):
    #    isskin[vertices_list.index(col)] = 0 # int((col == cloth_colors).all(axis=1).any())

    #should_work = should_work * isskin

    bp_rm_v = remove_bodyparts(smpl,clothed_body,item["category_id"])
    should_work = should_work * bp_rm_v

    # keep bodyparts that should not be in the final mesh 
    bp_gd_v = keep_bodyparts(smpl,clothed_body,item["category_id"])

    should_work = should_work * isin 
    should_work = should_work + bp_gd_v
    should_work[should_work >= 1] = 1   

    # remove the front middle mesh that are not in the cloth
    should_work = should_work * (1- middle_vertices_to_remove)
    
    garment_mesh = mask_mesh_vertices(clothed_body,should_work)
    garment_mesh = keep_biggest_comp(garment_mesh)
    trimesh.smoothing.filter_humphrey(garment_mesh)
    garment_mesh = fill_holes(garment_mesh,num_edges=20)
    garment_mesh.visual.vertex_colors = [0,102,51,255]
    garment_mesh.export(os.path.join(garments_path,img_number+'_'+item_name+'.obj'))


def segment_inwear(item,item_name,mask_zeros,img_number,clothed_body,smpl):

    img = cv2.imread(os.path.join(images_path,img_number+'.jpg'))
    item_segm = item['segmentation']
    expanded_clothes_mask = get_shape_mask(mask_zeros, item_segm, 15)

    clothed_body_colors_mask = color_vertices(img_number,expanded_clothes_mask,clothed_body)
    clothed_body_colors_mask  = (clothed_body_colors_mask[:,0] > 240)
    clothed_body_colors_v = -np.ones_like(clothed_body.visual.vertex_colors.view(np.ndarray))[:,:3]
    clothed_body_colors_v[clothed_body_colors_mask]  = clothed_body.visual.vertex_colors[clothed_body_colors_mask][:,:3]

    downpanded_clothes_mask = get_shape_mask(mask_zeros, item_segm, -5)
    downpandedclothed_body_colors = color_vertices(img_number,downpanded_clothes_mask,clothed_body)
    downpandedclothed_body_colors_v  = (downpandedclothed_body_colors[:,0] > 240)
    downpandedclothed_body_colors_v_col = clothed_body.visual.vertex_colors[downpandedclothed_body_colors_v][:,:3]

    cloth_colors = np.unique(downpandedclothed_body_colors_v_col,axis=0)

    isin = np.zeros((clothed_body_colors_v.shape[0],),dtype=np.int32)
    for i,col in enumerate(clothed_body_colors_v):
        isin[i] = int((col == cloth_colors).all(axis=1).any())
    
    should_work =  clothed_body_colors_mask #& isin#np.isin(clothed_body_colors_v, cloth_colors).all(axis=1)


    bp_gd_v = keep_bodyparts(smpl,clothed_body,item["category_id"])

    should_work = should_work * isin 


    should_work = should_work + bp_gd_v
    should_work[should_work >= 1] = 1   
    #skin_vertices = get_skin_vertices(folder_path,img_number,img, clothed_body)
    #skin_vertices = skin_vertices.tolist()
    #isskin = np.ones((clothed_body.vertices.shape[0],),dtype=np.int32)
    #vertices_list = clothed_body.vertices.tolist()
    #for i,col in enumerate(skin_vertices):
    #    isskin[vertices_list.index(col)] = 0 # int((col == cloth_colors).all(axis=1).any())
    #should_work = should_work * isskin

    garment_mesh = mask_mesh_vertices(clothed_body,should_work)
        
    bp_rm_v = remove_bodyparts(smpl,garment_mesh,item["category_id"])
    garment_mesh = mask_mesh_vertices(garment_mesh,bp_rm_v)


    trimesh.smoothing.filter_humphrey(garment_mesh)

    filled_garment_mesh = fill_holes(garment_mesh,num_edges=300)

    garment_mesh = remove_filling(filled_garment_mesh,garment_mesh,smpl,item["category_id"])

    garment_mesh = keep_biggest_comp(garment_mesh)
    garment_mesh.visual.vertex_colors = [0,102,51,255]
    garment_mesh.export(os.path.join(garments_path,img_number+'_'+item_name+'.obj'))



def segment_garment(img_number):
    with open(os.path.join(annos_path,img_number+'.json')) as json_file:
        segs = json.load(json_file)
    img = cv2.imread(os.path.join(images_path,img_number+'.jpg'))

    print(segs)
    mask_zeros = np.zeros_like(img, dtype='uint8')

    if not os.path.isfile(os.path.join(meshes_path,img_number+'_refine.obj')):
        print('Couldnt find ICON outputs for the image',img_number)
        return False
    clothed_body = trimesh.load((os.path.join(meshes_path,img_number+'_refine.obj')),maintain_order=True,process=False)

    smpl = trimesh.load((os.path.join(meshes_path,img_number+'_smpl.obj')),maintain_order=True,process=False)
    
    body_seg = np.load(os.path.join(segs_path,img_number+'.npy'))
    for item in segs.keys():
        if 'item' not in item:
            continue
        

        if overlap_cloth_body(body_seg,get_shape_mask(np.zeros_like(img), segs[item]['segmentation'],expansion=0)) < 0.8:
            print('Skipped cloth because not on body')
            continue
        print('Doing item',item)
        if segs[item]["category_id"] in [3,4]:# clothes not full in front, with zippers etc
            segment_outwear(segs[item],item,mask_zeros,img_number,clothed_body,smpl)

        else:
            segment_inwear(segs[item],item,mask_zeros,img_number,clothed_body,smpl)

    return True
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder_path', type=str)
    parser.add_argument('--img_number', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--div', type=int)

    args = parser.parse_args()

    annos_path = os.path.join(args.folder_path,'annos')
    segs_path = os.path.join(args.folder_path,'segs')
    images_path = os.path.join(args.folder_path,'images')
    meshes_path = os.path.join(args.folder_path,'meshes')
    garments_path = os.path.join(args.folder_path,'garments')

    if not os.path.isdir(garments_path):
        os.mkdir(garments_path)
    if args.test:
        print('TESTING SEGMENTATION ON ',args.img_number)
        segment_garment(args.img_number)
    else:
        print('RUNNING SEGMENTATION ON BIG DATA')
        united_done_name = os.path.join(args.folder_path,'united_img_number_done')
        imgs_done_div_name = os.path.join(args.folder_path,'img_divs/img_number_done_div' + str(args.div))
        imgs_to_do_div_name = os.path.join(args.folder_path,'img_divs/img_number_to_do_div' + str(args.div))
        with open(united_done_name, "rb") as fp:
            previous_imgs_done = pickle.load(fp)

        with open(imgs_to_do_div_name, "rb") as fp:
            imgs_to_do = pickle.load(fp)

        with open(imgs_done_div_name, "rb") as fp:
            imgs_done = pickle.load(fp)
        print('Have already done ',len(previous_imgs_done))      

        for img_number in tqdm.tqdm(imgs_to_do):
            if img_number not in previous_imgs_done:
                print('Doing image ', img_number)
                segmented_garmented = segment_garment(img_number)
                if not segmented_garmented:
                    continue
                imgs_done.append(img_number)
                with open(imgs_done_div_name, "wb") as fp:   
                    pickle.dump(imgs_done, fp)
