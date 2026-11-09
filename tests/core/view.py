import numpy as np, sys
from PIL import Image
def tm(x, exp=0.0):
    x = np.clip(x * 2**exp, 0, None)
    x = x / (1 + x) * 1.6  # soft
    x = np.clip(x, 0, 1) ** (1/2.2)
    return (x * 255 + 0.5).astype(np.uint8)
def save(arr, path, exp=0.0, scale=1):
    if arr.ndim == 2: arr = np.repeat(arr[..., None], 3, 2)
    im = Image.fromarray(tm(arr, exp))
    if scale != 1: im = im.resize((im.width*scale, im.height*scale), Image.NEAREST)
    im.save(path)
def strip(arrs, path, exp=0.0, crop=None, scale=1):
    ims = []
    for a in arrs:
        if a.ndim == 2: a = np.repeat(a[..., None], 3, 2)
        if crop: y0,y1,x0,x1 = crop; a = a[y0:y1, x0:x1]
        ims.append(tm(a, exp))
    im = Image.fromarray(np.concatenate(ims, 1))
    if scale != 1: im = im.resize((im.width*scale, im.height*scale), Image.NEAREST)
    im.save(path)
if __name__ == '__main__':
    a = np.load(sys.argv[1]); save(a, sys.argv[2], float(sys.argv[3]) if len(sys.argv)>3 else 0)
