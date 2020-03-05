"""
Visualize the projections in published HO-3D dataset
"""
from os.path import join
import pip
import argparse
from utils.vis_utils import *
import random
import numpy as np
from collections import OrderedDict
import logging

def install(package):
	if hasattr(pip, 'main'):
		pip.main(['install', package])
	else:
		from pip._internal.main import main as pipmain
		pipmain(['install', package])

try:
	import matplotlib.pyplot as plt
except:
	install('matplotlib')
	import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons

try:
	import chumpy as ch
except:
	install('chumpy')
	import chumpy as ch

try:
	import pickle
except:
	install('pickle')
	import pickle

import cv2
from mpl_toolkits.mplot3d import Axes3D

MANO_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), './mano/models/MANO_RIGHT.pkl')

# mapping of joints from MANO model order to simple order(thumb to pinky finger)
jointsMapManoToSimple = [0,
						 13, 14, 15, 16,
						 1, 2, 3, 17,
						 4, 5, 6, 18,
						 10, 11, 12, 19,
						 7, 8, 9, 20]

if not os.path.exists(MANO_MODEL_PATH):
	raise Exception('MANO model missing! Please run setup_mano.py to setup mano folder')
else:
	from mano.webuser.smpl_handpca_wrapper_HAND_only import load_model



class Constraints():
	def __init__(self):
		self.thetas = self.get_thetas()
		validThetaIDs, invalidThetaIDs, minThetaVals, maxThetaVals = self.parseThetaDescription(self.thetas)
		self.validThetaIDs = validThetaIDs
		self.invalidThetaIDs = invalidThetaIDs
		self.minThetaVals = minThetaVals
		self.maxThetaVals = maxThetaVals
		

	def get_thetas(self, unconstrained=False):
		MINBOUND = -5.		
		MAXBOUND = 5.
		thetas = OrderedDict([
			('global_rot_0', 	{'id': 0,  'min': MINBOUND,	'max': MAXBOUND,	'optimize': True}),
			('global_rot_1', 	{'id': 1,  'min': MINBOUND,	'max': MAXBOUND,	'optimize': True}),
			('global_rot_2', 	{'id': 2,  'min': MINBOUND,	'max': MAXBOUND,	'optimize': True}),

			('index_1_roll', 	{'id': 3,  'min': 0,		'max': 0.45,		'optimize': True}),
			('index_1_yaw', 	{'id': 4,  'min': -0.15,	'max': 0.2,		'optimize': True}),
			('index_1_pitch',	{'id': 5,  'min': 0.1,		'max': 1.8,		'optimize': True}),

			('index_2_roll', 	{'id': 6,  'min': -0.3,		'max': 0.2,		'optimize': True}),
			('index_2_yaw', 	{'id': 7,  'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('index_2_pitch',	{'id': 8,  'min': -0.0,		'max': 2.0,		'optimize': True}),

			('index_3_roll', 	{'id': 9,  'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('index_3_yaw', 	{'id': 10, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('index_3_pitch', 	{'id': 11, 'min': 0,		'max': 1.25,		'optimize': True}),


			('middle_1_roll', 	{'id': 12, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('middle_1_yaw', 	{'id': 13, 'min': -0.15,	'max': 0.15,		'optimize': True}),
			('middle_1_pitch', 	{'id': 14, 'min': 0.1,		'max': 2.0,		'optimize': True}),

			('middle_2_roll', 	{'id': 15, 'min': -0.5,		'max': -0.2,		'optimize': True}),
			('middle_2_yaw', 	{'id': 16, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('middle_2_pitch', 	{'id': 17, 'min': -0.0,		'max': 2.0,		'optimize': True}),

			('middle_3_roll', 	{'id': 18, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('middle_3_yaw', 	{'id': 19, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('middle_3_pitch', 	{'id': 20, 'min': 0,		'max': 1.25,		'optimize': True}),



			('pinky_1_roll', 	{'id': 21, 'min': -1.5,		'max': -0.2,		'optimize': True}),
			('pinky_1_yaw', 	{'id': 22, 'min': -0.15,	'max': 0.6,		'optimize': True}),
			('pinky_1_pitch', 	{'id': 23, 'min': -0.1,		'max': 1.6,		'optimize': True}),

			('pinky_2_roll', 	{'id': 24, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('pinky_2_yaw', 	{'id': 25, 'min': -0.5,		'max': 0.6,		'optimize': True}),
			('pinky_2_pitch', 	{'id': 26, 'min': -0.0,		'max': 2.0,		'optimize': True}),

			('pinky_3_roll', 	{'id': 27, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('pinky_3_yaw', 	{'id': 28, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('pinky_3_pitch', 	{'id': 29, 'min': 0,		'max': 1.25,		'optimize': True}),


			('ring_1_roll', 	{'id': 30, 'min': -0.5,		'max': -0.4,		'optimize': True}),
			('ring_1_yaw', 		{'id': 31, 'min': -0.25,	'max': 0.10,		'optimize': True}),
			('ring_1_pitch', 	{'id': 32, 'min': 0.1,		'max': 1.8,		'optimize': True}),

			('ring_2_roll', 	{'id': 33, 'min': -0.4,		'max': -0.2,		'optimize': True}),
			('ring_2_yaw', 		{'id': 34, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('ring_2_pitch', 	{'id': 35, 'min': -0.0,		'max': 2.0,		'optimize': True}),

			('ring_3_roll', 	{'id': 36, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('ring_3_yaw', 		{'id': 37, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('ring_3_pitch', 	{'id': 38, 'min': 0,		'max': 1.25,		'optimize': True}),


			('thumb_1_roll', 	{'id': 39, 'min': 0.0,		'max': 2.0,		'optimize': True}),
			('thumb_1_yaw', 	{'id': 40, 'min': -0.83,	'max': 0.66,		'optimize': True}),
			('thumb_1_pitch', 	{'id': 41, 'min': -0.0,		'max': 0.5,		'optimize': True}),

			('thumb_2_roll', 	{'id': 42, 'min': -0.15,	'max': 1.6,		'optimize': True}),
			('thumb_2_pitch', 	{'id': 43, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('thumb_2_yaw', 	{'id': 44, 'min': 0,		'max': 0.5,		'optimize': True}),

			('thumb_3_roll', 	{'id': 45, 'min': MINBOUND,	'max': MAXBOUND,	'optimize': False}),
			('thumb_3_pitch', 	{'id': 46, 'min': -0.5,		'max': 0,		'optimize': True}),
			('thumb_3_yaw', 	{'id': 47, 'min': -1.57,	'max': 1.08,		'optimize': True})
		])

		
		# change suggestions by analysis lw
		# thetas['pinky_2_yaw']['optimize'] = True
		thetas['pinky_2_yaw']['min'] = 0
		thetas['pinky_2_yaw']['max'] = 0.25

		# values beyond 1.5 are hard to reach with a real pinky
		thetas['pinky_2_pitch']['max'] = 1.5
		
		# because ring_3 axis doesn't allow for simple curling the finer tip by pitch adjustment without also adjusting yaw
		thetas['ring_3_yaw']['min'] = -0.4
		thetas['ring_3_yaw']['max']= 0.0
		thetas['ring_3_yaw']['optimize'] = True


		thetas['thumb_2_pitch']['optimize'] = True
		thetas['thumb_2_pitch']['min']= -0.3
		thetas['thumb_2_pitch']['max']= 0.3

		# 
		thetas['thumb_3_pitch']['min'] = -1.5
		# thetas['thumb_3_pitch']['max']= 1.5
		thetas['thumb_3_pitch']['max']= 0.9 # should suffice also for the lego builders thumb

		thetas['thumb_3_yaw']['min'] = -0.4
		thetas['thumb_3_yaw']['max']= 0.0
		


		## config for non-extreme thumb pose
		thetas['thumb_3_pitch']['max']= 0.1 # if thumb is not firmly pressed against an object
		thetas['thumb_3_yaw']['min'] = -0.2

		if unconstrained:
			for theta in thetas.values():
				theta['min'] = min(theta['min'], -2.0) #MINBOUND
				theta['max'] = max(theta['max'], 2.0) #MAXBOUND
				theta['optimize'] = True
		return thetas
		

	def parseThetaDescription(self, thetas):
		
		validThetaIDs = []
		invalidThetaIDs = []
		minThetaVals = np.zeros(len(thetas))
		maxThetaVals = np.zeros(len(thetas))
		for theta in thetas.values():
			thetaID = theta['id']
			minThetaVals[thetaID] = theta['min']
			maxThetaVals[thetaID] = theta['max']
			if theta['optimize']:
				validThetaIDs.append(thetaID)
			else:
				invalidThetaIDs.append(thetaID)

		validThetaIDs = np.array(validThetaIDs, dtype=np.int32)
		invalidThetaIDs = np.array(invalidThetaIDs, dtype=np.int32)

		return validThetaIDs, invalidThetaIDs, minThetaVals, maxThetaVals




def forwardKinematics(fullpose, trans, beta):
	'''
	MANO parameters --> 3D pts, mesh
	:param fullpose:
	:param trans:
	:param beta:
	:return: 3D pts of size (21,3)
	'''

	assert fullpose.shape == (48,)
	assert trans.shape == (3,)
	assert beta.shape == (10,)

	m = load_model(MANO_MODEL_PATH, ncomps=6, flat_hand_mean=True)
	m.fullpose[:] = fullpose
	m.trans[:] = trans
	m.betas[:] = beta

	return m.J_transformed.r, m

def add_pose_slider(theta_name, theta, column_index, row_index, x_offset=0.2, y_offset=0.3):
	pose_index = theta['id']
	slider_active = theta['optimize']
	slider_color = 'blue' if slider_active else 'gray'
	slider_background = 'white' if slider_active else 'lightgray'
	ax_pose = plt.axes([x_offset + 0.16 * column_index, y_offset - 0.035 * row_index, 0.075, 0.015], facecolor=slider_background)

	slider_initial_val = np.clip(anno['handPose'][pose_index], theta['min'], theta['max'])
	pose_slider = Slider(ax_pose, theta_name, theta['min'], theta['max'], 
						valinit=slider_initial_val, valstep=0.01, color=slider_color)
	pose_slider.set_active(slider_active)
	return pose_index, pose_slider

if __name__ == '__main__':

	# parse the input arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("ho3d_path", nargs='?', default=None, help="Path to HO3D dataset")
	# ap.add_argument("ycbModels_path", type=str, help="Path to ycb models directory")
	ap.add_argument("-split", required=False, type=str,
					help="split type", choices=['train', 'evaluation'], default='train')
	ap.add_argument("-seq", required=False, type=str,
					help="sequence name")
	ap.add_argument("-id", required=False, type=str,
					help="image ID")
	ap.add_argument("-visType", required=False,
					help="Type of visualization", choices=['matplotlib'], default='matplotlib') #choices=['open3d', 'matplotlib']
	ap.add_argument("-unconstrained", action="store_true", default=False,
					help="use mano constraints")
	args = vars(ap.parse_args())

	# YCBModelsDir = args['ycbModels_path']
	
	# some checks to decide if visualizing default pose or specific ho3d sample
	if not all([args['ho3d_path'], args['seq'], args['id'], args['split']]):
		if any([args['ho3d_path'], args['seq'], args['id'], args['split']]):
			logging.warn('Not all parameters for HO3D set. Visualizing default pose.')
		anno = {
			'handPose': np.zeros((48)),
			'handTrans': np.zeros((3)),
			'handBeta': np.zeros((10))
		}	
	else:
		anno = read_annotation(args['ho3d_path'], args['seq'], args['id'], args['split'])

	constraints = Constraints()
	thetas = constraints.get_thetas(unconstrained=args['unconstrained'])

	# Visualize
	if args['visType'] == 'matplotlib':

		fig, ax = plt.subplots()
		plt.subplots_adjust(bottom=0.35)

		handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])

		# show 3D hand mesh
		ax2 = fig.add_subplot(1, 2, 1, projection="3d")
		plot3dVisualize(ax2, handMesh, flip_x=False, isOpenGLCoords=True, c="viridis", elev_azim=(90,-90))

		ax3 = fig.add_subplot(1, 2, 2, projection="3d")
		plot3dVisualize(ax3, handMesh, flip_x=False, isOpenGLCoords=True, c="viridis", elev_azim=(0,-90))

		pose_sliders = []

		def update(val):
			for pose_index, slider in pose_sliders:
				anno['handPose'][pose_index] = slider.val
			drawHandPose(anno)
				
		def drawHandPose(anno):				
			ax2.clear()
			ax3.clear()
			handJoints3D, handMesh = forwardKinematics(anno['handPose'], anno['handTrans'], anno['handBeta'])
			plot3dVisualize(ax2, handMesh, flip_x=False, isOpenGLCoords=True, c='viridis', elev_azim=None)
			plot3dVisualize(ax3, handMesh, flip_x=False, isOpenGLCoords=True, c='viridis', elev_azim=None)
			fig.canvas.draw_idle()


		for index, theta_description  in enumerate(thetas.items()[:3]):
			theta_name, theta = theta_description
			column_index = index // 9
			row_index = index - 9 * column_index
			pose_index, pose_slider = add_pose_slider(theta_name, theta, column_index, row_index, x_offset=0.05)
			pose_slider.on_changed(update)
			pose_sliders.append((pose_index, pose_slider))

		for index, theta_description  in enumerate(thetas.items()[3:]):
			theta_name, theta = theta_description
			column_index = index // 9
			row_index = index - 9 * column_index
			pose_index, pose_slider = add_pose_slider(theta_name, theta, column_index, row_index)
			pose_slider.on_changed(update)
			pose_sliders.append((pose_index, pose_slider))

		plt.show()
	else:
		raise Exception('Unknown visualization type')

