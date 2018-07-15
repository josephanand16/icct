import os
import cv2
import io
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob
from PIL import Image
from itertools import tee



def load_image( infilename ) :
	img = Image.open( infilename )
	img.load()
	data = np.asarray( img, dtype="float32" )
	return data

def save_image( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save( outfilename )

def image_write_rgb(filename,output_path,outputarray):
	cv2.imwrite(os.path.normpath(output_path)+'/' + filename + 'rgb.jpg', outputarray[...,::-1])

def image_write_bgr(filename,output_path,outputarray):
	cv2.imwrite(os.path.normpath(output_path)+'/' + filename + 'bgr.jpg', outputarray)

def image_write_gbr(filename,output_path,outputarray):
	cv2.imwrite(os.path.normpath(output_path)+'/' + filename + 'gbr.jpg', outputarray[...,[2,0,1]])

def snp(image,tuner):

	row,col,ch = image.shape
	s_vs_p = 0.5 + tuner*0.1
	amount = 0.04 + tuner*0.01
	out = np.copy(image)
	# Salt mode
	num_salt = np.ceil(amount * image.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))
		  for i in image.shape]
	out[coords] = 1

	# Pepper mode
	num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))
		  for i in image.shape]
	out[coords] = 0
	return out

def poisson(image):

	vals = len(np.unique(image))
	vals = 2 ** np.ceil(np.log2(vals))
	noisy = np.random.poisson(image * vals) / float(vals)
	return noisy

def speckle(image):

	row,col,ch = image.shape
	gauss = np.random.randn(row,col,ch)
	gauss = gauss.reshape(row,col,ch)        
	noisy = image + image * gauss
	return noisy

def commonAugCorrupted(inputpath,outputdir,classification):
	
	output_path = outputdir + '/' + classification
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		fliplr=iaa.Sequential([iaa.Fliplr(0.5)]).augment_image(img)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplr_%d.jpg'%(classification,j) , fliplr[...,::-1])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplr_gbr_%d.jpg'%(classification,j) , fliplr[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplr_bgr_%d.jpg'%(classification,j) , fliplr)
		fliplrud = iaa.Sequential([iaa.Flipud(0.5)]).augment_image(fliplr)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplrud_%d.jpg' %(classification,j) , fliplrud[...,::-1])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplrud_gbr_%d.jpg' %(classification,j) , fliplrud[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplrud_bgr_%d.jpg' %(classification,j) , fliplrud)
		flipud = iaa.Sequential([iaa.Flipud(0.5)]).augment_image(img)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_flipud_%d.jpg' %(classification,j) , flipud[...,::-1])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_flipud_gbr_%d.jpg' %(classification,j) , flipud[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_flipud_bgr_%d.jpg' %(classification,j) , flipud)
		for img1 in (fliplr,fliplrud,flipud,img):
			for i in range(10):
				rot = iaa.Sequential([iaa.Affine(rotate=(-25+0.1*i,25+0.1*i))]).augment_image(img1)
				cv2.imwrite(os.path.normpath(output_path)+'/%s_rotate_%d%d.jpg'%(classification,j,i) , rot[...,::-1])
				cv2.imwrite(os.path.normpath(output_path)+'/%s_rotate_%d%d_gbr.jpg'%(classification,j,i) , rot[...,[2,0,1]])
				cv2.imwrite(os.path.normpath(output_path)+'/%s_rotate_bgr_%d%d.jpg'%(classification,j,i) , rot)
		j = j + 1

def commonAug(inputpath,outputdir,classification):
	
	output_path = outputdir + '/' + classification
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		fliplr=iaa.Sequential([iaa.Fliplr(0.5)]).augment_image(img)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplr_gbr_%d.jpg'%(classification,j) , fliplr[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplr_bgr_%d.jpg'%(classification,j) , fliplr)
		fliplrud = iaa.Sequential([iaa.Flipud(0.5)]).augment_image(fliplr)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplrud_gbr_%d.jpg' %(classification,j) , fliplrud[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_fliplrud_bgr_%d.jpg' %(classification,j) , fliplrud)
		flipud = iaa.Sequential([iaa.Flipud(0.5)]).augment_image(img)
		cv2.imwrite(os.path.normpath(output_path)+'/%s_flipud_gbr_%d.jpg' %(classification,j) , flipud[...,[2,0,1]])
		cv2.imwrite(os.path.normpath(output_path)+'/%s_flipud_bgr_%d.jpg' %(classification,j) , flipud)
		for img1 in (fliplr,fliplrud,flipud,img):
			for i in range(10):
				rot = iaa.Sequential([iaa.Affine(rotate=(-25+0.1*i,25+0.1*i))]).augment_image(img1)
				cv2.imwrite(os.path.normpath(output_path)+'/%s_rotate_%d%d.jpg'%(classification,j,i) , rot[...,[2,0,1]])
				cv2.imwrite(os.path.normpath(output_path)+'/%s_rotate_bgr_%d%d.jpg'%(classification,j,i) , rot)
		j = j + 1

def saltPepperNoise(inputpath,outputdir):

	output_path = outputdir + '/' + 'saltPepper'
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		for k in range(5):
			outputnoisy = snp(img,k)
			image_write_gbr('saltPepper_%d%d' %(j,k),output_path,outputnoisy)
			image_write_rgb('saltPepper_%d%d' %(j,k),output_path,outputnoisy)
			image_write_bgr('saltPepper_%d%d' %(j,k),output_path,outputnoisy)
		j = j + 1


def gaussNoise(inputpath,outputdir):

	output_path = outputdir + '/' + 'gaussNoise'
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		for k in range(5):
			outputnoisy = gaussian(img,mean=0.1*k)
			image_write_gbr('gauss_%d%d' %(j,k),output_path,outputnoisy)
			image_write_rgb('gauss_%d%d' %(j,k),output_path,outputnoisy)
			image_write_bgr('gauss_%d%d' %(j,k),output_path,outputnoisy)
		j = j + 1

def pairwise(iterable):
    """Awesome function from the itertools cookbook
    https://docs.python.org/2/library/itertools.html
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_header_length(byteArray):

	for i, pair in enumerate(pairwise(byteArray)):
		if pair[0] == 255 and pair[1] == 218:
			result = i + 2
			return result

	raise ValueError('Not a valid jpg!')

def glitch_bytes(image_bytes,amount,seed,iterations):

	amount = amount / 100
	seed = seed / 100
	new_bytes = copy.copy(image_bytes)
	header_length = get_header_length(image_bytes)

	for i in (range(iterations)):
		max_index = len(image_bytes) - header_length - 4
		px_min = int((max_index / iterations) * i)
		px_max = int((max_index / iterations) * (i + 1))
		delta = (px_max - px_min)  # * 0.8
		px_i = int(px_min + (delta * seed))
		if (px_i > max_index):
			px_i = max_index

		byte_index = header_length + px_i
		new_bytes[byte_index] = int(amount * 256)

	return new_bytes

def speckleNoise(inputpath,outputdir):

	output_path = outputdir + '/' + 'speckle'
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		outputnoisy = speckle(img)
		image_write_gbr('speckle_%d' %(j),output_path,outputnoisy)
		image_write_rgb('speckle_%d' %(j),output_path,outputnoisy)
		image_write_bgr('speckle_%d' %(j),output_path,outputnoisy)
		j = j + 1

def poissonNoise(inputpath,outputdir):

	output_path = outputdir + '/' + 'poisson'
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		img = load_image(os.path.normpath(image))
		outputnoisy = poisson(img)
		image_write_gbr('poisson_%d' %(j),output_path,outputnoisy)
		image_write_rgb('poisson_%d' %(j),output_path,outputnoisy)
		image_write_bgr('poisson_%d' %(j),output_path,outputnoisy)
		j = j + 1

def imageGlitch(inputpath,outputdir,amount,seed,iterations):

	output_path = outputdir + '/' + 'glitch'
	try:
		os.mkdir(output_path)
	except:
		pass
	j=1
	for image in inputpath:
		f = open(image, mode='rb')
		image_bytes = bytearray(f.read())
		outputnoisy = glitch_bytes(image_bytes,amount,seed,iterations)
		stream = io.BytesIO(outputnoisy)
		im = Image.open(stream)
		im.save(os.path.normpath(output_path)+'/glitched_%d_%d.jpg' %(j,amount))
		j = j + 1

inputlocation = 'C:/icct/image/sd_corrupt_original'
inputlocation = 'C:/icct/image/snp_original'
outputdir = 'C:/icct/image/output'
classification = 'sd_corrupt'
path = glob.glob(inputlocation+"/*")


for i in range(30):
	imageGlitch(path,'C:/icct/image/output',i,7,6)
