'''
Author: Justin Chen
Date: 	2.15.2020
'''
import os
import sys
import time
import atexit
import psutil
import platform
from torch import cuda
from datetime import date
from pymongo import MongoClient
from bson.objectid import ObjectId

class Record(dict):
	'''
	inputs	host (string)	MongoDB host name
			port (int)		MongoDB port number
	'''
	def __init__(self, _id='', host='localhost', port=27017, database='experiments', collection='parameters'):
		super().__init__()

		# Start MongoDB daemon
		stream = os.popen('mongod')

		# Connect to MongoDB
		self.client = MongoClient(host, port)
		self.database = database
		self.collection = collection
		self.db = self.client[database]
		self.col = self.db[collection]
		self._id = str(_id) if isinstance(_id, ObjectId) else _id

		if len(self._id) > 0: self.update(self.col.find_one({'_id': ObjectId(self._id)}))

		self.system_info()


	'''
	Update a dictionary value. Useful for tracking multiple experiments
	e.g {
		'model_0': {'time': 1606588095.295271, 'lr': 0.01},
		'model_1': {'time': 1606588103.197889, 'lr': 0.001}
	}

	update('model_1', {'time': 1606588162.7220979, 'lr': 0.002})

	{
		'model_0': {'time': 1606588095.295271, 'lr': 0.01},
		'model_1': {'time': 1606588162.7220979, 'lr': 0.002}
	}

	inputs	value (any value)	     Value of parameter
			key   (string, optional) Key to dict stored object
	'''
	def update(self, value, key=None):

		if self.is_argparse(value):
			value = vars(value)
		
		if not key is None:
			if type(key) is not str:
				raise Exception('Key parameter must be type String.')

			if key in self:
				self[key].update(value)
			else:
				self[key] = value
		else:
			super().update(value)


	'''
	Check if argument is argparse.Namespace

	inputs:  args (*) 	 	Variable
	outputs: res  (bool)	True if args is an instance of argparse.Namespace, else False
	'''
	def is_argparse(self, args):
		try:
			return args.__module__ == 'argparse'
		except AttributeError:
			return False


	'''
	Save the record to experiment document

 	output: id (string) Inserted document id
	'''
	def save(self):
		doc_id = self.col.insert_one(dict(self.items())).inserted_id
		if len(self._id) == 0: self._id = doc_id


	'''
	Save system information
	'''
	def system_info(self):
		uname = platform.uname()
		gpus = [cuda.get_device_name(i) for i in range(cuda.device_count())]


		self.update({
			'python': platform.python_version(),
			'machine': uname.machine,
			'processor': uname.processor,
			'os': os.name,
			'os_name': platform.system(),
			'os_ver': platform.release(),
			'memory': str(psutil.virtual_memory().total//2**30)+' GB',
			'storage': str(psutil.disk_usage('/').total//2**30)+' GB',
			'user': os.getlogin(),
			'gpus': gpus
		})
	