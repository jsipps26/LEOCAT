
import numpy as np
import os, sys

from leocat.utils.time import date_to_jd, jd_to_date, jd_to_date_vec, \
								date_to_ymd
#
from leocat.utils.index import hash_value, hash_index
from leocat.utils.general import format_as_array


class GeoLayers():
	def __init__(self, layer_list, quality_function, prob=True):
		if not isinstance(layer_list, (list, tuple)):
			layer_list = [layer_list]
		self.layer_list = layer_list
		self.quality_function = quality_function
		self.prob = prob

	def get_quality(self, x, y, JD): #, *layers):
		layers = self.layer_list
		return self.quality_function(x, y, JD, *layers)


class AnalyticLayer():
	def __init__(self, analytic_function, name=None):
		self.analytic_function = analytic_function
		self.name = None

	def get_value(self, x, y, JD):
		return self.analytic_function(x, y, JD)


class RasterLayer():
	"""

	Can make a code that reads a directory
	and ensures all requirements are met.
		code makes some metadata file?

	name = os.path.basename(directory)
	filetype implied as constant
		files = os.listdir(directory)
		filetype = files.split('.')[-1]
		ensure filetypes are all the same

	projection is 4326 LLA
	res should be on dataset
	xlim, ylim should be on dataset
	JD_lim should be extractable from dataset
	res_JD should be extractable from dataset
	x_origin, y_origin..?
		Is this for the dataset or coverage grid?
		for the dataset
		can be derived from xlim, ylim

	JD_origin can be derived from datasets
	remove download_function

	read_function required
	quality_function required
	attrs required
	remove force_year
		offload this to filenaming before processing

	parser optional

	file must exist
	attrs:
	List that contains preset time identifiers:
		year, month, day, doy
	Each has a preset character length:
		4, 2, 2, 3
		zfill required
		January starts at 1
		doy starts at 1
	
	Instead of specifying attributes, derive them from filename

	Time fields are separated by a single parser
		parser can be an empty string
	
	keys must be one of:
		year, month, day, hour, minute, second
	and they must be sorted
	if a "smaller" time-span is present, the larger must also be present
	i.e. cannot just specify a month or day

	"""
	def __init__(self, file, keys, read_function, x_edges=[-180.0,180.0], y_edges=[-90.0,90.0], \
					parser='', name=None):
	#

		if not os.path.exists(file):
			raise FileNotFoundError('%s not found' % file)

		file_base = os.path.basename(file)
		file_ext = os.path.splitext(file_base)[1]
		L_ext = len(file_ext)

		keys_lower = []
		for entry in keys:
			keys_lower.append(entry.lower())
		keys = keys_lower

		keys_to_num_char = {'year': 4, 'month': 2, 'day': 2, \
							'hour': 2, 'minute': 2, 'second': 2}
		#

		num_char = 0
		for i,key in enumerate(keys):
			num_char = num_char + keys_to_num_char[key]
		num_char = num_char + i*len(parser)
		# prefix = prefix[:-num_char]
		prefix = file_base[:-(num_char+L_ext)]
		postfix = file_base[-(num_char+L_ext):-L_ext]

		if parser == '':
			value = []
			i = 0
			for key in keys:
				num = keys_to_num_char[key]
				try:
					val = int(postfix[i:(i+num)])
				except ValueError:
					raise ValueError('file keys must be numeric (1)')
				value.append(val)
				i += num
			value = tuple(value)

		else:
			try:
				value = tuple([int(val) for val in postfix.split(parser)])
			except ValueError:
				raise ValueError('file keys must be numeric (2)')

		self.file = file
		self.file_base = file_base
		self.file_ext = file_ext
		self.prefix = prefix
		self.parser = parser

		self.read_function = read_function

		# keys are like year, month, day
		# value is like 2024, 1, 1
		#	{'year': 2024, 'month': 1, 'day': 1}
		self.keys = keys
		self.value = value

		# derive res, origin from dataset
		#	can do at load_data command
		# zz = read_function(file)
		self.data = None
		self.name = name
		self.space_params = {'x_edges': x_edges, 'y_edges': y_edges}

		date1 = list(value)
		date2 = list(value)
		date2[-1] = date2[-1] + 1

		date1_ymd = date_to_ymd(date1)
		date2_ymd = date_to_ymd(date2)

		JD1 = date_to_jd(*date1_ymd)
		JD2 = date_to_jd(*date2_ymd)

		self.time_params = {'JD1': JD1, 'JD2': JD2}


	def set_data(self):
		data = self.read_function(self.file)
		self.data = data
		self.set_space_params()
		self.set_time_params()


	def get_data(self, xy=False):
		# data assumed in column-major
		#	t, x, y or x, y
		# i.e. zz[x,y] instead of zz[y,x]
		#	could maybe make a setting for this
		data = self.data
		if data is None:
			self.set_data()
		data = self.data

		if xy:
			zz = data
			space_params = self.space_params
			x_origin, y_origin = space_params['x_origin'], space_params['y_origin']
			res_x, res_y = space_params['res_x'], space_params['res_y']

			if len(zz.shape) == 2:
				num_cols, num_rows = zz.shape
			elif len(zz.shape) == 3:
				_, num_cols, num_rows = zz.shape

			# make xx, yy
			x = hash_value(np.arange(num_cols), x_origin, res_x)
			y = hash_value(np.arange(num_rows), y_origin, res_y)
			xx, yy = np.meshgrid(x,y)
			xx, yy = xx.T, yy.T

			return xx, yy, zz

		else:
			return data


	def set_space_params(self):

		space_params = self.space_params
		zz = self.data

		if len(zz.shape) == 2:
			num_cols, num_rows = zz.shape
		elif len(zz.shape) == 3:
			_, num_cols, num_rows = zz.shape

		x_edges = space_params['x_edges']
		y_edges = space_params['y_edges']

		x_origin = x_edges[0]
		y_origin = y_edges[0]
		res_x = (x_edges[1]-x_edges[0]) / num_cols
		res_y = (y_edges[1]-y_edges[0]) / num_rows

		space_params['x_origin'] = x_origin
		space_params['y_origin'] = y_origin
		space_params['res_x'] = res_x
		space_params['res_y'] = res_y

		self.space_params = space_params


	def set_time_params(self):

		time_params = self.time_params
		zz = self.data

		is_temporal = False
		if len(zz.shape) == 3:
			is_temporal = True
		time_params['is_temporal'] = is_temporal

		if is_temporal:
			JD1, JD2 = time_params['JD1'], time_params['JD2']
			JD_edges = [JD1,JD2]
			num_JD, num_cols, num_rows = zz.shape
			JD_origin = JD_edges[0]
			res_JD = (JD_edges[1]-JD_edges[0]) / num_JD

			time_params['JD_origin'] = JD_origin
			time_params['res_JD'] = res_JD


	def get_value(self, x, y, JD):

		zz = self.data
		if zz is None:
			zz = self.get_data()

		x = format_as_array(x)
		y = format_as_array(y)
		JD = format_as_array(JD)

		q_layer = np.full(JD.shape,np.nan)
		space_params = self.space_params
		time_params = self.time_params

		JD1, JD2 = time_params['JD1'], time_params['JD2']

		is_temporal = time_params['is_temporal']
		if is_temporal:
			JD_origin = time_params['JD_origin']
			res_JD = time_params['res_JD']

		b = (JD >= JD1) & (JD < JD2)
		if b.sum() == 0:
			return q_layer

		x_origin, y_origin = space_params['x_origin'], space_params['y_origin']
		res_x, res_y = space_params['res_x'], space_params['res_y']
		cols = hash_index(x[b], x_origin, res_x)
		rows = hash_index(y[b], y_origin, res_y)

		if is_temporal:
			k_vec = hash_index(JD[b], JD_origin, res_JD)
			q_layer[b] = zz[k_vec,cols,rows]
		else:
			q_layer[b] = zz[cols,rows]

		return q_layer


class RasterLayerSeries():
	# all keys must be the same
	# each raster layer must be temporally disjoint from others?
	def __init__(self, layer_list, name=None):
		layers = {}
		for layer in layer_list:
			layers[layer.value] = layer
		self.layers = layers
		self.name = name

		# Check that keys are all the same
		keys_count = {}
		for value in layers:
			layer = layers[value]
			keys_layer = layer.keys
			for key in keys_layer:
				if not (key in keys_count):
					keys_count[key] = 0
				keys_count[key] += 1

		err = 0
		for key in keys_count:
			count = keys_count[key]
			if not (count == len(layers)):
				err = 1

		if err:
			raise Exception('keys must be the same for every layer in series')

		self.keys = layer.keys # assuming keys are the same for every layer



	def get_value(self, x, y, JD):

		x = format_as_array(x)
		y = format_as_array(y)
		JD = format_as_array(JD)

		layers = self.layers
		keys = self.keys

		dates = jd_to_date_vec(JD)
		dates = dates.astype(int)
		if len(dates.shape) == 1:
			dates = np.array([dates])
		dates_uq = np.unique(dates, axis=0)

		q_layer_series = np.full(JD.shape,np.nan)

		for (year,month,day) in dates_uq:
			# Query the layer corresponding to (y,m,d)
			attrs = {}
			for key in keys:
				if key == 'year':
					attrs[key] = year
				elif key == 'month':
					attrs[key] = month
				elif key == 'day':
					attrs[key] = day
			#
			value = tuple(attrs.values())
			try:
				layer = layers[value]
			except KeyError:
				continue

			JD_lower, JD_upper = layer.time_params['JD1'], layer.time_params['JD2']
			b = (JD >= JD_lower) & (JD < JD_upper)
			if b.sum() == 0:
				continue

			q_layer = layer.get_value(x[b], y[b], JD[b])
			q_layer_series[b] = q_layer

		return q_layer_series
