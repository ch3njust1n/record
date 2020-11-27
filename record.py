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

class Record(object):
	'''
	inputs	host (string)	MongoDB host name
			port (int)		MongoDB port number
	'''
	def __init__(self, host='localhost', port=27017, database='experiments', collection='parameters'):
		
		# Start MongoDB daemon
		stream = os.popen('mongod')

		# Connect to MongoDB
		self.client = MongoClient(host, port)
		self._id = None # Get record id
		self.database = database
		self.collection = collection
		self.db = self.client[database]
		self.col = self.db[collection]

		self.record = {}

		self.system_info()

		# Save before process ends
		atexit.register(self.save)


	'''
	Add experiment parameters to experiment record.

	inputs	key   (string)	Name of parameter
			value (any)		Value of parameter
	'''
	def add(self, key, value):
		if type(key) is not str:
			raise Exception('Key parameter must be type String.')

		if self.is_argparse(value):
			value = vars(value)
		
		self.record[key] = value


	'''
	Update record with dictionary

	inputs: obj (dict or argparse.Namespace)	Dictionary or argparse.Namespace with values to update record
	'''
	def extend(self, obj):
		is_argparse = self.is_argparse(obj)

		if type(obj) is not dict and not is_argparse:
			raise Exception('Input must be a dictionary or argparse.Namespace')

		if is_argparse:
			obj = vars(obj)

		self.record.update(obj)


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
		doc_id = self.col.insert_one(self.record).inserted_id
		return doc_id


	'''
	Save system information
	'''
	def system_info(self):
		uname = platform.uname()
		gpus = [cuda.get_device_name(i) for i in range(cuda.device_count())]


		self.extend({
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


	'''
	Retrieve record

	inputs:  record_id (string)	MongoDB document _id
	outputs: record    (dict)   Dictionary of record corresponding to record_id
	'''
	def get(self, record_id):
		return self.col.find_one({'_id': ObjectId(record_id)})


