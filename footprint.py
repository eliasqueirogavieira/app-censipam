import os
import rpcm
from osgeo import gdal
from shapely.geometry import Point, Polygon

import h5py
from rpcm import RPCModel


import geopandas


def calculate_h5_footprint(file):

	f = h5py.File(file, 'r')
	rpcs = f['RPC']
	rpc_data = {}
	for rpc_key in rpcs.keys():
		if rpcs[rpc_key].ndim > 0:
			rpc_data[rpc_key] = str(rpcs[rpc_key][:].tolist()).replace(',','').replace('[','').replace(']','')
		else:
			rpc_data[rpc_key] = str(rpcs[rpc_key][()])
	Himg, Wimg = f['s_i'].shape
	Zimg = rpcs['HEIGHT_OFF'][()]
 
	rpc_calc = RPCModel(rpc_data)
	footprint = [rpc_calc.localization(0, 0, Zimg), rpc_calc.localization(Wimg-1, 0, Zimg),
		rpc_calc.localization(Wimg-1, Himg-1, Zimg), rpc_calc.localization(0, Himg-1, Zimg)]
	return footprint


def calculate_footprint(cur_file):

	height = os.popen("gdalinfo " + cur_file + " | grep HEIGHT_OFF | sed 's/HEIGHT_OFF=//'").read()
	if len(height)>0:
		height = float(height)
	else:
		height = None

	try:

		footprint = rpcm.image_footprint(cur_file, height, False)
		footprint = footprint["geometry"]["coordinates"][0]

	except:
		d = gdal.Open(cur_file)
		if (d.GetGCPCount() > 0):
			GT = gdal.GCPsToGeoTransform(d.GetGCPs())
		else:
			GT = d.GetGeoTransform()
		
		footprint = []
		for X_pixel in [0, d.RasterXSize-1]:
			for Y_line in [0, d.RasterYSize-1]:
				footprint.append([
					GT[0] + X_pixel * GT[1] + Y_line * GT[2],
					GT[3] + X_pixel * GT[4] + Y_line * GT[5]])
	return footprint


def foot2shp(footprint):

	polygon   = Polygon([(pts[0],pts[1]) for pts in footprint])
	bb = {'bl': [polygon.bounds[0], polygon.bounds[1]],
		'br': [polygon.bounds[2], polygon.bounds[1]],
		'ur': [polygon.bounds[2], polygon.bounds[3]],
		'ul': [polygon.bounds[0], polygon.bounds[3]]}
	
	t = Polygon([ bb['bl'], # left, bottom
		bb['br'], # right, bottom
		bb['ur'], # right, top
		bb['ul'] ]) # left, top

	di = di = {'item': ["footprint"], "geometry": [t]}
	gp = geopandas.GeoDataFrame(di, crs = "EPSG:4326")
	gp.to_file('file.shp')

def foot2polygon(footprint):

	polygon   = Polygon([(pts[0],pts[1]) for pts in footprint])
	bb = {'bl': [polygon.bounds[0], polygon.bounds[1]],
		'br': [polygon.bounds[2], polygon.bounds[1]],
		'ur': [polygon.bounds[2], polygon.bounds[3]],
		'ul': [polygon.bounds[0], polygon.bounds[3]]}
	
	t = Polygon([ bb['bl'], # left, bottom
		bb['br'], # right, bottom
		bb['ur'], # right, top
		bb['ul'] ]) # left, top

	return t


def find_tiff_intersections_ref(tif_file, tif_fp, other_tifs, other_fps):
	"""_summary_ returns intersected files

	Args:
		tif_file (_type_): _description_
		tif_fp (_type_): _description_
		other_tifs (_type_): _description_
		other_fps (_type_): _description_

	Returns:
		_type_: _description_
	"""

	intersect_file_names = []
	for cur_file, cur_footprint in zip(other_tifs, other_fps):

		if cur_file != tif_file: # Avoid testing the original file
			#cur_polygon   = Polygon([(pts[0], pts[1]) for pts in cur_footprint])
			cur_polygon = foot2polygon(cur_footprint)
			ok = 0
			for pts in tif_fp:
				cur_pt = Point(pts[0],pts[1])
				if cur_pt.within(cur_polygon):
					intersect_file_names.append(cur_file)
					ok +=1
					break
			if ok==0:
				cur_polygon = Polygon([(pts[0],pts[1]) for pts in tif_fp])
				for pts in cur_footprint:
					cur_pt = Point(pts[0],pts[1])
					if cur_pt.within(cur_polygon):
						intersect_file_names.append(cur_file)
						break
	return intersect_file_names



def find_tiff_intersections(tif_fp, other_fps):
	"""_summary_ returns intersected files

	Args:
		tif_file (_type_): _description_
		tif_fp (_type_): _description_
		other_tifs (_type_): _description_
		other_fps (_type_): _description_

	Returns:
		_type_: _description_
	"""

	#cur_polygon   = Polygon([(pts[0], pts[1]) for pts in cur_footprint])
	cur_polygon = foot2polygon(other_fps)

	for pts in tif_fp:
		cur_pt = Point(pts[0],pts[1])
		if cur_pt.within(cur_polygon):
			return True

	return False
