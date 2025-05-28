import math
import random
import torch
import cv2
from PIL import Image
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.ndimage import binary_dilation,binary_fill_holes
from skimage.morphology import disk

def MaskGenerator(image_size, batch_size):
    masks = torch.zeros((batch_size,1,image_size,image_size),dtype=torch.int64)  
    for i in range(batch_size):
        img = np.random.rand(image_size, image_size)
        seed_r = np.random.randint(0, image_size)
        seed_c = np.random.randint(0, image_size)
        img[seed_r, seed_c] = 2
        
        iter_count = 2
        n = np.random.randint(800, 3000) # 500,3000
        while iter_count < n:
            iter_count += 1
            minr = max(0, seed_r - 1)
            maxr = min(255, seed_r + 1)
            minc = max(0, seed_c - 1)
            maxc = min(255, seed_c + 1)
            win = img[minr:maxr+1, minc:maxc+1]
            minw = np.min(win)
            win[win == minw] = iter_count
            img[minr:maxr+1, minc:maxc+1] = win
            seed_positions = np.argwhere(img == iter_count)
            if len(seed_positions) > 0:
                seed_r, seed_c = seed_positions[0]
                
        mask = img >= 2
        random_structure = disk(np.random.randint(4, 6)) 
        mask = binary_dilation(mask, structure=random_structure)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        #se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, se)
        filled_image = binary_fill_holes(mask)
        filled_image = filled_image.astype(np.int64)
        filled_image = filled_image[np.newaxis,np.newaxis,:]#.astype(np.int64)
        filled_image = torch.from_numpy(filled_image)

        masks[i,:,:,:]=filled_image

    return masks

def load_data(
    *,
    A_dir,
    mask_dir="",
    B_dir="",
    batch_size,
    image_size,
    rank=0,
    world_size=1,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):

    if not A_dir:
        raise ValueError("unspecified image A directory")
    if not mask_dir:
        raise ValueError("unspecified mask directory")
    if not B_dir:
        raise ValueError("unspecified image B directory")
    all_A = _list_image_files_recursively(A_dir)
    all_masks = _list_image_files_recursively(mask_dir)
    all_B = _list_image_files_recursively(B_dir)
    dataset = ImageDataset(
        image_size,
        all_A,
        all_masks,
        all_B,
        shard=rank,
        num_shards=world_size,
        random_crop=random_crop,
        random_flip=random_flip,
    )

    #return dataset
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    #return loader
    while True:
        yield from loader

def load_conditions(
    *,
    A_dir,
    B_dir,
    mask_dir,
    image_size,
):
    if not A_dir:
        raise ValueError("unspecified image A directory")
    if not B_dir:
        raise ValueError("unspecified image B directory")
    all_A = _list_image_files_recursively(A_dir)
    all_B = _list_image_files_recursively(B_dir)
    all_masks = None
    if mask_dir != "":
        all_masks = _list_image_files_recursively(mask_dir)
    dataset = GuidedmapDataset(all_A, all_B, all_masks, image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def load_single_image(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    # arr = np.array(pil_image)
    arr = center_crop_arr(pil_image, 256)
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    return arr
    
class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_A_paths,
        mask_paths,
        image_B_paths,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images_A = image_A_paths[shard:][::num_shards]
        self.local_images_B = image_B_paths[shard:][::num_shards]
        self.local_masks = mask_paths[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images_B)

    def __getitem__(self, idx):
        pathA = self.local_images_A[idx]
        with bf.BlobFile(pathA, "rb") as A:
            pil_image_A = Image.open(A)
            pil_image_A.load()
        pil_image_A = pil_image_A.convert("RGB")

        pathB = self.local_images_B[idx]
        with bf.BlobFile(pathB, "rb") as B:
            pil_image_B = Image.open(B)
            pil_image_B.load()
        pil_image_B = pil_image_B.convert("RGB")

        path_mask = self.local_masks[idx]
        with bf.BlobFile(path_mask, "rb") as f:
            pil_mask = Image.open(f)
            pil_mask.load()
        pil_mask = pil_mask.convert("L")

        if self.random_crop:
            arrB = random_crop_arr(pil_image_B, self.resolution)
            arrA = random_crop_arr(pil_image_A, self.resolution)
            arrm = random_crop_arr(pil_mask, self.resolution)
        else:
            arrB = center_crop_arr(pil_image_B, self.resolution)
            arrA = center_crop_arr(pil_image_A, self.resolution)
            arrm = center_crop_arr(pil_mask, self.resolution)
        if self.random_flip and random.random() < 0.5:
            arrB = arrB[:, ::-1]
            arrA = arrA[:, ::-1]
            arrm = arrm[:, ::-1]
        arrA = arrA.astype(np.float32) / 127.5 - 1
        arrB = arrB.astype(np.float32) / 127.5 - 1
        arrm[arrm>0] = 1
        arrm = arrm[np.newaxis,:].astype(np.int64)

        out_dict = {}
        out_dict["mask"] = arrm
        return np.transpose(arrB, [2, 0, 1]), np.transpose(arrA, [2, 0, 1]), out_dict
    
class GuidedmapDataset(Dataset):
    def __init__(self, imageA, imageB, masks, resolution):
        self.imageA = imageA
        self.imageB = imageB
        self.masks = masks
        self.resolution = resolution

    def __len__(self):
        return len(self.imageA)

    def __getitem__(self, idx):
        path_imageA = self.imageA[idx]
        with bf.BlobFile(path_imageA, "rb") as f:
            pil_imageA = Image.open(f)
            pil_imageA.load()

        path_imageB = self.imageB[idx]
        with bf.BlobFile(path_imageB, "rb") as s:
            pil_imageB = Image.open(s)
            pil_imageB.load()
        
        pil_imageA = pil_imageA.convert("RGB")
        pil_imageB = pil_imageB.convert("RGB")

        arrA = center_crop_arr(pil_imageA, self.resolution)

        arrB = center_crop_arr(pil_imageB, self.resolution)

        arrA = arrA.astype(np.float32) / 127.5 - 1
        
        if self.masks is not None:
            path_mask = self.masks[idx]

            with bf.BlobFile(path_mask, "rb") as m:
                pil_mask = Image.open(m)
                pil_mask.load()
            
            pil_mask = pil_mask.convert("L")
            arrm = center_crop_arr(pil_mask, self.resolution)
            arrm[arrm>0] = 1
            arrm = arrm[np.newaxis,:].astype(np.int64)
            
            return np.transpose(arrA, [2, 0, 1]),np.transpose(arrB, [2, 0, 1]), arrm
        else:
            return np.transpose(arrA, [2, 0, 1]),np.transpose(arrB, [2, 0, 1])




def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]