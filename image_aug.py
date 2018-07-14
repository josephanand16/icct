import os
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob



def load_image( infilename ) :
	img = Image.open( infilename )
	img.load()
	data = np.asarray( img, dtype="float32" )
	return data

def save_image( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save( outfilename )

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


inputlocation = 'C:/icct/image/sd_corrupt_original'
outputdir = 'C:/icct/image/output'
classification = 'sd_corrupt'
path = glob.glob(inputlocation+"/*")
commonAugCorrupted(path,'C:/icct/image/output',classification)

# location = 'C:/icct/image/sd_corrupt_original'
# outputdir = 'C:/icct/image/output'
# classification = 'sd_uncorrupt'
# path = glob.glob(location)
# commonAugCorrupted(path,'C:/icct/image/output',classification)
