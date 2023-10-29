"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
#from pytorch3d.renderer.mesh import rasterize_meshes
#from pytorch3d.structures import Meshes

import torch.nn as nn


def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': True,
            'cull_backfaces': True,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(),
                               faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1],
                                     3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat(
            [pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals

def get_visibility(xy, z, faces):
    """get the visibility of vertices

    Args:
        xy (torch.tensor): [N,2]
        z (torch.tensor): [N,1]
        faces (torch.tensor): [N,3]
        size (int): resolution of rendered image
    """

    xyz = torch.cat((xy, -z), dim=1)
    xyz = (xyz + 1.0) / 2.0
    faces = faces.long()

    rasterizer = Pytorch3dRasterizer(image_size=2**12)
    meshes_screen = Meshes(verts=xyz[None, ...], faces=faces[None, ...])
    raster_settings = rasterizer.raster_settings

    pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
        meshes_screen,
        image_size=raster_settings.image_size,
        blur_radius=raster_settings.blur_radius,
        faces_per_pixel=raster_settings.faces_per_pixel,
        bin_size=raster_settings.bin_size,
        max_faces_per_bin=raster_settings.max_faces_per_bin,
        perspective_correct=raster_settings.perspective_correct,
        cull_backfaces=raster_settings.cull_backfaces,
    )

    vis_vertices_id = torch.unique(faces[torch.unique(pix_to_face), :])
    vis_mask = torch.zeros(size=(z.shape[0], 1))
    vis_mask[vis_vertices_id] = 1.0

    # print("------------------------\n")
    # print(f"keep points : {vis_mask.sum()/len(vis_mask)}")

    return vis_mask


def query_color_face(verts, faces, image, device):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = faces.float().to(device).split([2, 1], dim=1)
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]])
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = torch.nn.functional.grid_sample(
        image, uv, align_corners=True)  # [B, C, N, 1]
    colors = ((colors[0, :, :, 0].permute(1, 0) + 1.0) * 0.5 *
              255.0).detach().cpu()# * visibility

    # mesh = trimesh.Trimesh(verts.detach().cpu(), faces.detach().cpu(), process=False, maintains_order=True)
    # mesh.visual.vertex_colors = colors
    # mesh.show()

    return colors

def query_color_vis(verts, faces, image, device):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = verts.split([2, 1], dim=1)
    visibility = get_visibility(xy, z, faces[:, [0, 2, 1]])
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = torch.nn.functional.grid_sample(
        image, uv, align_corners=True,mode='nearest')  # [B, C, N, 1]
    colors = colors.detach().cpu() * visibility

    # mesh = trimesh.Trimesh(verts.detach().cpu(), faces.detach().cpu(), process=False, maintains_order=True)
    # mesh.visual.vertex_colors = colors
    # mesh.show()

    return colors


def query_color_no_visibility(verts, faces, image, device):
    """query colors from points and image

    Args:
        verts ([B, 3]): [query verts]
        faces ([M, 3]): [query faces]
        image ([B, 3, H, W]): [full image]

    Returns:
        [np.float]: [return colors]
    """

    verts = verts.float().to(device)
    faces = faces.long().to(device)

    (xy, z) = verts.split([2, 1], dim=1)
    uv = xy.unsqueeze(0).unsqueeze(2)  # [B, N, 2]
    uv = uv * torch.tensor([1.0, -1.0]).type_as(uv)
    colors = torch.nn.functional.grid_sample(
        image, uv, align_corners=True,mode='nearest')  # [B, C, N, 1]
    colors = colors.detach().cpu() #* visibility

    # mesh = trimesh.Trimesh(verts.detach().cpu(), faces.detach().cpu(), process=False, maintains_order=True)
    # mesh.visual.vertex_colors = colors
    # mesh.show()

    return colors
