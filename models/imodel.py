
from abc import ABC, abstractmethod
from osgeo import gdal, ogr, osr

def pred2Shapefile(in_fname, out_fname):
	# https://gis.stackexchange.com/questions/417383/how-to-apply-gdal-polygonize
	#  get raster datasource
	#gdal_polygonize.py -8 crop_2_8.tif shape.shp

	src_ds = gdal.Open( in_fname )
    
	srcband = src_ds.GetRasterBand(1)

	dst_layername = 'deforestation'
	drv = ogr.GetDriverByName("ESRI Shapefile")
	dst_ds = drv.CreateDataSource(out_fname)

	sp_ref = osr.SpatialReference()
	sp_ref.SetFromUserInput('EPSG:4326')

	dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )

	fld = ogr.FieldDefn("HA", ogr.OFTInteger)
	dst_layer.CreateField(fld)
	dst_field = dst_layer.GetLayerDefn().GetFieldIndex("HA")

	options = []
	options.append('8CONNECTED=8')

	gdal.Polygonize( srcband, srcband.GetMaskBand(), dst_layer, dst_field, [], callback=None )


class NNModel(ABC):
	@abstractmethod
	def load(self):
		pass

	@abstractmethod
	def predict_single(self):
		pass