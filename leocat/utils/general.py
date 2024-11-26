
import os, sys
import numpy as np
from leocat.utils.const import *
from leocat.utils.time import ymdhms_to_val, date_to_jd
from leocat.utils.index import overlap


def format_as_array(vec):
	if isinstance(vec, np.ndarray):
		return vec
	elif isinstance(vec, (int, float)):
		return np.array([vec])
	elif isinstance(vec, (list, tuple)):
		return np.array(vec)

		
def search_leocat(search_string, verbose=1, directory=MAIN_DIR):

	def find_string_in_file(file_path, search_string):
		"""Check if the search string is in the file."""
		with open(file_path, 'r') as file:
			for line_no, line in enumerate(file, start=1):
				if search_string in line:
					file_path_last = os.path.join(*file_path.split(os.sep)[-2:-1])
					file_path_base = os.path.basename(file_path)
					print(f"Found in {os.path.join(file_path_last, file_path_base)} at line {line_no}")
					# print(f"Found in {os.path.basename(file_path)} at line {line_no}")
					if verbose > 1:
						print('  ', line)
					# print(f"Found in {file_path} at line {line_no}")

	"""Recursively search for the string in all files in the given directory."""
	for root, dirs, files in os.walk(directory):
		for file in files:
			if file.endswith('.py'):  # Assuming we are only interested in Python files
				file_path = os.path.join(root, file)
				if verbose > 2:
					print(file_path)
				try:
					find_string_in_file(file_path, search_string)
				except UnicodeDecodeError:
					# This can happen if trying to open non-text files
					pass

	
def write_pickle(filename, data, complex_object=True):

	def write_dill(data,fp):
		import dill
		dill.settings['resurse'] = True
		dill.dump(data,fp)

	import pickle
	with open(filename,'wb') as fp:
		if complex_object:
			write_dill(data,fp)
		else:
			try:
				pickle.dump(data,fp)
			except pickle.PicklingError:
				write_dill(data,fp)
			except AttributeError:
				write_dill(data,fp)

def read_pickle(filename, complex_object=True):

	def read_dill(fp):
		import dill
		dill.settings['resurse'] = True
		data = dill.load(fp)
		return data

	import pickle
	with open(filename, 'rb') as fp:
		if complex_object:
			data = read_dill(fp)
		else:
			try:
				data = pickle.load(fp)
			except pickle.PicklingError:
				data = read_dill(fp)
			except AttributeError:
				data = read_dill(fp)

	return data


def rnp():
	# remove numpy printing (rnp)
	np.set_printoptions(suppress=True)

def pause(msg='enter to continue'):
	yn = input(msg)
	if yn != '':
		sys.exit()
