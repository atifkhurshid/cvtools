from .io import imshow, imread, imwrite
from .processing import imresize, imresize_maximum, imscale

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 'webp', '.ppm', '.pgm', '.pbm', '.pnm', '.hdr', '.pic', '.dib']
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])
