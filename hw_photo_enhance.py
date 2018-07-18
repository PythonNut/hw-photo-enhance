import numpy as np
import argparse
from pathlib import Path
from scipy import ndimage, stats
from PIL import Image, ImageEnhance, ImageFilter

parser = argparse.ArgumentParser(
    description='Process a photo to match a black-on-white document.'
)

parser.add_argument('img')
fname = Path(parser.parse_args().img)

I = ndimage.imread(str(fname))
I = np.ndarray.astype(I,'float64')

def blur(I, n):
    im = Image.fromarray(np.ndarray.astype(I, 'uint8'))
    im = im.filter(ImageFilter.GaussianBlur(n))
    im = ImageEnhance.Sharpness(im).enhance(2)
    return np.ndarray.astype(np.asarray(im), 'float64')

B = blur(I,50)
Z = I + (np.mean(B) -B)

F = np.ndarray.flatten(Z)
D = stats.kde.gaussian_kde(F)
D.covariance_factor = lambda: 0.75
D._compute_covariance()
X = np.linspace(np.min(F), np.max(F), 100)
Y = D(X)
min_point, max_point, *_ = [X[i] for i in range(1,len(Y)-1) if Y[i] > Y[i-1] and Y[i] > Y[i+1]]

split = 0.1

low_point  = (1 - split) * min_point + split * max_point
high_point = split * min_point + (1 - split) * max_point
Z -= low_point
Z *= 255/(high_point-low_point)
# Z -= np.percentile(np.ndarray.flatten(Z), 2)
# Z *= 255/np.percentile(np.ndarray.flatten(Z), 2.5)
print(Z.shape)
if Z.shape[2] == 4:
    Z[:,:,3]=255
Z = np.clip(Z, 0, 255)
Q = Image.fromarray(np.ndarray.astype(Z, 'uint8'))

result = Q
# result =ImageEnhance.Contrast(ImageEnhance.Brightness(Q).enhance(1.5)).enhance(3)
result.show()
result.save(Path(fname.stem + "-contrast").with_suffix(fname.suffix))
