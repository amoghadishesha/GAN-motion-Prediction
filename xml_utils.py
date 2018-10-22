from xml.dom import minidom
import numpy as np
import pdb
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy.stats import multivariate_normal

def xml_parsing(file_name, p1_id, p2_id, total_joint = 18):

	doc = minidom.parse(file_name)
	Joint = doc.getElementsByTagName("Joint")
	ske = doc.getElementsByTagName("Skeleton")
	id_info = []
	
	for i in ske:
		id_info.append(int(i.getElementsByTagName('PlayerId')[0].firstChild.data))
	assert len(id_info) == 6
	assert p1_id in id_info
	assert p2_id in id_info
	Joint_info_p1 = {}
	Joint_info_p2 = {}
	counter = 0
	for j in Joint:
		joint_name = j.getElementsByTagName("JointType")[0].firstChild.data
		if 'Hand' in joint_name:
			continue
		Pos_3d = j.getElementsByTagName("Position")[0]
		x = float(Pos_3d.getElementsByTagName("X")[0].firstChild.data)
		y = float(Pos_3d.getElementsByTagName("Y")[0].firstChild.data)
		z = float(Pos_3d.getElementsByTagName("Z")[0].firstChild.data)
		state = j.getElementsByTagName("TrackingState")[0].firstChild.data
		if counter <  total_joint:
			Joint_info_p1[joint_name] = [x,y,z,state]
		else:
			Joint_info_p2[joint_name] = [x,y,z,state]
		counter += 1
	if id_info.index(p1_id) < id_info.index(p2_id):
		return Joint_info_p1,Joint_info_p2
	else:
		return Joint_info_p2,Joint_info_p1

def get_structure_info():
	name_codebook = [u'HipCenter', u'Head', u'HipLeft', u'KneeRight', u'ShoulderRight', \
				u'Spine', u'WristRight', u'AnkleLeft', u'KneeLeft', u'ElbowLeft', \
				u'ShoulderCenter', u'FootRight', u'WristLeft', u'HipRight', u'FootLeft', \
				u'ElbowRight',  u'AnkleRight', u'ShoulderLeft']
	joint_num = len(name_codebook)
	bones = np.zeros((joint_num,joint_num))
	bones[name_codebook.index('HipCenter'),name_codebook.index('Spine')] = 1
	bones[name_codebook.index('HipCenter'),name_codebook.index('HipLeft')] = 1
	bones[name_codebook.index('HipCenter'),name_codebook.index('HipRight')] = 1
	bones[name_codebook.index('Spine'),name_codebook.index('ShoulderCenter')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('ShoulderRight')] = 1
	bones[name_codebook.index('ShoulderRight'),name_codebook.index('ElbowRight')] = 1
	bones[name_codebook.index('ElbowRight'),name_codebook.index('WristRight')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('ShoulderLeft')] = 1
	bones[name_codebook.index('ShoulderLeft'),name_codebook.index('ElbowLeft')] = 1
	bones[name_codebook.index('ElbowLeft'),name_codebook.index('WristLeft')] = 1
	bones[name_codebook.index('ShoulderCenter'),name_codebook.index('Head')] = 1
	bones[name_codebook.index('HipRight'),name_codebook.index('KneeRight')] = 1
	bones[name_codebook.index('KneeRight'),name_codebook.index('AnkleRight')] = 1
	bones[name_codebook.index('AnkleRight'),name_codebook.index('FootRight')] = 1
	bones[name_codebook.index('HipLeft'),name_codebook.index('KneeLeft')] = 1
	bones[name_codebook.index('KneeLeft'),name_codebook.index('AnkleLeft')] = 1
	bones[name_codebook.index('AnkleLeft'),name_codebook.index('FootLeft')] = 1
	bones = bones + bones.T
	return name_codebook,bones

def plot_3d(pose_3d_dict):
	name_codebook, bones = get_structure_info()
	joint_num = len(pose_3d_dict.keys())
	x = np.zeros((joint_num,))
	y = np.zeros((joint_num,))
	z = np.zeros((joint_num,))
	
	for i in range(joint_num):
		pose_3d_coor = pose_3d_dict[name_codebook[i]]
		x[i] = pose_3d_coor[0]
		y[i] = pose_3d_coor[1]
		z[i] = pose_3d_coor[2]

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for i in range(joint_num):
		ax.scatter(x[i], y[i], z[i])

	for i in range(joint_num):
		for j in range(joint_num):
			if bones[i,j] == 1:
				ax.plot([x[i],x[j]], [y[i],y[j]], [z[i],z[j]])
	max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
	ax.plot(x, z, 'r+', zdir='y', zs=1.5)
	mid_x = (x.max()+x.min()) * 0.5
	mid_y = (y.max()+y.min()) * 0.5
	mid_z = (z.max()+z.min()) * 0.5
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.set_xlim(mid_x - max_range, mid_x + max_range)
	ax.set_ylim(mid_y - max_range, mid_y + max_range)
	ax.set_zlim(mid_z - max_range, mid_z + max_range)
	plt.show()

def get_pose_numpy_array(pose_3d_dict, get_2d = False):
	joint_num = len(pose_3d_dict.keys())
	name_codebook, bones = get_structure_info()
	pose_3d = np.zeros((joint_num,3))
	counter = 0
	for key in name_codebook:
		pose_3d[counter,:] = pose_3d_dict[key][:3]
		counter += 1
	if get_2d:
		pose_3d = pose_3d[:,[1,2]]
	return pose_3d

def get_gaussian_gt_3d(pose_3d,sigma, grid_point = 64,pad_space = 1):
	z1 = np.min(pose_3d[:,2])
	z2 = z1 + 2*pad_space
	x1 = (pose_3d[11,0] + pose_3d[14,0])/2 -pad_space
	x2 = (pose_3d[11,0] + pose_3d[14,0])/2 + pad_space
	y1 = -pad_space
	y2 = pad_space
	coor = [x1,x2,y1,y2,z1,z2]
	x_line = np.linspace(x1,x2,grid_point, endpoint = True)
	y_line = np.linspace(y1,y2,grid_point, endpoint = True)
	z_line = np.linspace(z1,z2,grid_point, endpoint = True)
	x_mesh,y_mesh,z_mesh = np.meshgrid(x_line,y_line,z_line)
	pos = np.empty((x_mesh.size,3))
	pos[:,0] = np.reshape(x_mesh,[-1])
	pos[:,1] = np.reshape(y_mesh,[-1])
	pos[:,2] = np.reshape(z_mesh,[-1])
	joint_num = pose_3d.shape[0]
	gt_map = np.zeros((grid_point,grid_point,grid_point))

	for i in range(joint_num):
		pdf = multivariate_normal(pose_3d[i,:], [[sigma, 0,0], [0,sigma,0], [0, 0, sigma]])
		gt_map += np.reshape(pdf.pdf(pos),[grid_point,grid_point,grid_point])
	return gt_map/np.max(gt_map),coor

def visual_3d_gt_pose(s, threshold, coor, grid_point = 32):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x_line = np.linspace(coor[0],coor[1],grid_point, endpoint = True)
	y_line = np.linspace(coor[2],coor[3],grid_point, endpoint = True)
	z_line = np.linspace(coor[4],coor[5],grid_point, endpoint = True)
	x_mesh,y_mesh,z_mesh = np.meshgrid(x_line,y_line,z_line)
	s = np.reshape(s,[-1])
	x_mesh = np.reshape(x_mesh,[-1])
	y_mesh = np.reshape(y_mesh,[-1])
	z_mesh = np.reshape(z_mesh,[-1])
	colmap = cm.ScalarMappable(cmap=cm.hsv)
	colmap.set_array(s)
	yg = ax.scatter(x_mesh, y_mesh, z_mesh, c=cm.hsv(s/max(s)), marker='o')
	cb = fig.colorbar(colmap)
	plt.show()

def get_gaussian_gt(pose_2d,sigma,img_h,img_w, l_pad = 1, r_pad = 1, h_pad = 2):
	y1 = np.min(pose_2d[:,1])
	x1 = -l_pad
	x2 = r_pad
	y2 = y1 + h_pad
	coor = [x1,x2,y1,y2]
	x_line = np.linspace(x1,x2,img_w, endpoint = True)
	y_line = np.linspace(y1,y2,img_h, endpoint = True)
	x_mesh,y_mesh = np.meshgrid(x_line,y_line)
	pos = np.empty(x_mesh.shape+ (2,))
	pos[:,:,0] = x_mesh
	pos[:,:,1] = y_mesh
	joint_num = pose_2d.shape[0]
	gt_map = np.zeros((img_h,img_w))
	for i in range(joint_num):
		pdf = multivariate_normal(pose_2d[i,:], [[sigma, 0], [0, sigma]])
		gt_map += np.reshape(pdf.pdf(pos),[img_h,img_w])
	return gt_map/np.max(gt_map),coor

def visual_2d_gt_pose(s, coor, grid_point = 32):
	fig = plt.figure()
	x_line = np.linspace(coor[0],coor[1],grid_point, endpoint = True)
	y_line = np.linspace(coor[2],coor[3],grid_point, endpoint = True)
	x_mesh,y_mesh = np.meshgrid(x_line,y_line)
	s = np.reshape(s,[-1])
	x_mesh = np.reshape(x_mesh,[-1])
	y_mesh = np.reshape(y_mesh,[-1])
	colmap = cm.ScalarMappable(cmap=cm.hsv)
	colmap.set_array(s)
	yg = plt.scatter(x_mesh, y_mesh, c=cm.hsv(s/max(s)), marker='o')
	cb = fig.colorbar(colmap)
	plt.show()

if __name__ == '__main__':
	grid_point = 64
	pad_space = 1
	p1,p2 = xml_parsing('Skeleton 0.xml', 5,6)
	plot_3d(p1)
	# p1_2d = get_pose_numpy_array(p1, get_2d = True)
	# p1_2d,coor = get_gaussian_gt(p1_2d,0.01,grid_point,grid_point)
	# visual_2d_gt_pose(p1_2d,coor,grid_point)
	# pdf,coor= get_gaussian_gt_3d(p1_3d,0.001, grid_point = grid_point, pad_space = pad_space)
	# visual_3d_gt_pose(pdf,0.1, coor,grid_point = grid_point)