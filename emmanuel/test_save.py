import tensorflow as tf
import numpy as np
import keras
from keras.models import *
import nibabel as nib
from medpy.io import load,save
import sys
sys.path.append('../data_manipulation')
sys.path.append('../utils')
import glob

from losses import *
from metrics import *
from data_Load import *
from data_generator import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model_ensemble(folder_name):
	path = '../../my_weights/'+folder_name+'/64deepVal*'
	models_path = glob.glob(path)
	dice = []
	for paths in models_path:
		dice.append(paths[-5:-3])
	models_path_dict = dict(zip(models_path,dice))
	models_path_sorted = sorted(models_path_dict, key=models_path_dict.__getitem__)
	#return load_model(models_path_sorted[-1],custom_objects={'dice_coef_loss': dice_coef_loss,
	#		'sensitivity':sensitivity,'specificity':specificity})
	weights = []
	for i in range(1,7):
		model = load_model(models_path_sorted[-i],custom_objects={'dice_coef_loss': dice_coef_loss,
			'sensitivity':sensitivity,'specificity':specificity})
		weights.append(model.get_weights())
		#print 'weights shape: ',np.ndarray(model.get_weights()[0])
	
	weights_mean = np.mean(weights,axis=0)
	#print 'meaned weights shape: ',weights_mean.shape
	print 'loaded weights mean'
	model.set_weights(weights_mean)
	print 'loaded weights mean into model'

	return model
	
def predict_patch(path,batch_size,patch_size):	
	
	model = load_model_ensemble(path)
	
	print 'model loaded'
	my_loader= Load(batch_size=batch_size,patch_len=patch_size)
	testIm_patch,testSeg_patch,testIm,image_shape = my_loader.load_test()
	
	testSeg_patch = testSeg_patch.astype('float32')
	
	print("Predicting")
	output_Array = model.predict(testIm_patch,batch_size=batch_size)
	output_Array = output_Array.astype('float32')
	sess = tf.Session()
	print "Dice: ",sess.run(dice_coef(output_Array,testSeg_patch,0))
	print output_Array.shape, '  ',testSeg_patch.shape
	return output_Array,testSeg_patch,image_shape
	
def predict_image(path,batch_size,patch_size):	
	
	model = load_model(path,custom_objects={'dice_coef_loss': dice_coef_loss,
	'sensitivity':sensitivity,'specificity':specificity})
	
	my_loader= Load(batch_size=batch_size,patch_len=patch_size)
	
	testIm,testSeg = my_loader.loadImages(iD='test',cont = 1,downsampling=2,isotrope= True)

	print("Predicting")
	output_Array = model.predict(testIm,batch_size=batch_size)
	output_Array = output_Array[0].astype('float32')
	testSeg = testSeg[0].astype('float32')
	#sess = tf.Session()
	#print "Dice: ",sess.run(dice_coef(output_Array,testSeg,0))
	return output_Array,testSeg
	

def rebuild_image_unif(output_Array,patch_size,image_shape):
	print("array to image")
	import math
	image_shape = np.asarray(image_shape).astype('float16')
	a = math.ceil(image_shape[0]/patch_size)
	b = math.ceil(image_shape[1]/patch_size)
	c = math.ceil(image_shape[2]/patch_size)
	n_paches = np.array([a,b,c]).astype('uint16')
	image_shape = image_shape.astype('uint16')

	output_Image = np.zeros((image_shape[0],image_shape[1],image_shape[2],1), dtype=np.float32)
	print "n_patchs ",n_paches

	cont = 0
		
	dict_shape = {'i':0,'j':1,'k':2}
	print n_paches[0]
	for i in range(n_paches[0]):
		for j in range(n_paches[1]):
			for k in range(n_paches[2]):
				
				index = []
				index_patch = []
				dict_num = {'i':i,'j':j,'k':k}
				for a in ['i','j','k']:
					#print a
					index.append(dict_num[a]*patch_size) #index init
					
					if dict_num[a] == n_paches[dict_shape[a]]-1:
						index_patch.append(image_shape[dict_shape[a]]-(n_paches[dict_shape[a]]-1)*patch_size)
						index.append(None)
					else:
						index.append((dict_num[a]+1)*patch_size)
						index_patch.append(None)

				output_Image[index[0]:index[1],index[2]:index[3],index[4]:index[5]] = output_Array[cont,:index_patch[0],:index_patch[1],:index_patch[2]]
				cont += 1
	return output_Image
	
def rebuild_image_superporsed(output_Array):
	print("array to image")
	
	patch_len = 44
	out_patches_shape =  np.array([5, 5, 3])
	image_shape = patch_len*out_patches_shape
	output_Image = np.zeros((image_shape[0],image_shape[1],image_shape[2],1), dtype='float32')

	cont = 0
	for i in range(out_patches_shape[0]):
		for j in range(out_patches_shape[1]):
			for k in range(out_patches_shape[2]):
				output_Image[i*patch_len:(i+1)*patch_len,j*patch_len:(j+1)*patch_len,k*patch_len:(k+1)*patch_len] = output_Array[cont]
				cont += 1
				
	output_Image=np.pad(output_Image,2,'constant')
	output_Image = output_Image[:,:,4:132,2]
	print 'image_shape',output_Image.shape
	return output_Image		

def predict_build_patch(patch_size = 64):
	
	prediction,seg,a = predict_patch('KerasPatchVnetDeep5',batch_size = 4, patch_size=patch_size)
	Ims = rebuild_image_unif(prediction,patch_size=patch_size,image_shape=a)
	seg = rebuild_image_unif(seg,patch_size=patch_size,image_shape=a)
	save_nii(Ims,'../../Results/test30.nii')
	save_nii(seg,'../../Results/test30seg.nii')

def predict_build_image():
	patch_size=64
	prediction,seg = predict_image('KerasFullFcnDeep',batch_size = 1, patch_size=patch_size)
	save_nii(prediction,'../../Results/test25.nii')
	save_nii(seg,'../../Results/test25seg.nii')
	
def save_nii(output_Image,path):
	save(output_Image,path)

def test_rebuild():
	patch_size = 64
	my_loader= Load(batch_size=1,patch_len=patch_size)
	testIm_patch,testSeg_patch,testIm,image_shape = my_loader.load_test()
	#rebuildre
	Im = rebuild_image_unif(testIm_patch,patch_size,image_shape)
	
	testIm = testIm.astype('float32')
	Im = Im.astype('float32')

	save_nii(Im,'../../Results/testRebuild.nii')
	save_nii(testIm,'../../Results/testRebuild_True.nii')

def test_patch():
	my_loader= Load(batch_size=1,patch_len=32)
	
	testIm_patch,testSeg_patch,testIm,image_shape = my_loader.load_test()
	data = testIm_patch
	seg = testSeg_patch
	data = data.astype('float32')
	seg = seg.astype('float32')

	for i in range(len(data)):
		if seg[i].mean() > 0.005:
			seg1 = seg[i]
			print seg1.mean()
			print data.max()
			#seg1 = np.pad(seg1,10,'constant')
			#seg1 = seg1[:,:,:,10]
			save_nii(data[i],'dataPatch.nii')
			save_nii(seg[i],'segPatch.nii')
			return data, seg
			
def test_patch_generator():
	my_generator = Generator(patch_len=64 , batch_size=1)

	for batch_features, batch_labels in my_generator.generatorRandomPatchs('val'):
		if batch_labels[0].mean() > 0.005:
			save_nii(batch_features[0].astype('float32'),'dataPatch.nii')
			save_nii(batch_labels[0].astype('float32'),'segPatch.nii')
			break
if __name__ == '__main__':

	#test_patch_generator()
	test_rebuild()
	#test_patch()
	#predict_build_image()
	#predict_build_patch()
