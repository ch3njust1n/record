'''
Author: Justin Chen
Date: 	2.15.2020
'''
import os
import json
import atexit
import signal
import psutil
import platform
from torch import cuda
from pymongo import MongoClient
from bson.objectid import ObjectId

class Record(dict):
	'''
	inputs:	
	_id        (string, optional)  Record Mongo object id
	host       (string, optional)  MongoDB host name
	port       (int, optional)	   MongoDB port number
	database   (string, optional)  MongoDB database
	collection (string, optional)  MongoDB collection
	save_dir   (string, optional)  Save Record to file as well as MongoDB
	'''
	def __init__(self, _id='', host='localhost', port=27017, database='experiments', collection='parameters', save_dir=''):
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
		self.save_dir = save_dir

		if len(self._id) > 0: self.update(self.col.find_one({ '_id': ObjectId(self._id) }))

		self.system_info()

		atexit.register(self.save)
		signal.signal(signal.SIGTERM, self.save)
		signal.signal(signal.SIGINT, self.save)


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

	inputs:
	value (any value)        Value of parameter
	key   (string, optional) Key to dict stored object. Default: None
	'''
	def update(self, value, key=None):

		if self.is_argparse(value):
			value = vars(value)

		if self.is_configparser(value):
			value = { 'config': { k : dict(value[k].items()) for k, _ in value.items() } }

		if key:
			if not isinstance(key, str):
				raise Exception('Key parameter must be type String.')
				
			if key in self and isinstance(self[key], dict):
				self[key].update(value)
			else:
				self[key] = value
		else:
			super().update(value)


	'''
	Check if argument is argparse.Namespace

	inputs:  
	args (*) Variable

	outputs: 
	res  (bool)	True if args is an instance of argparse.Namespace, else False
	'''
	def is_argparse(self, args):
		try:
			return args.__module__ == 'argparse'
		except AttributeError:
			return False


	'''
	Check if argument is configparser.ConfigParser

	inputs:  
	args (*) Variable

	outputs: 
	res (bool) True if args is an instance of configparser.ConfigParser, else False
	'''
	def is_configparser(self, args):
		try:
			return args.__module__ == 'configparser'
		except:
			return False


	'''
	Save the record to experiment document

 	output: 
 	id (string) Inserted document id
	'''
	def save(self):
		record = dict(self.items())
		doc_id = self.col.insert_one(record).inserted_id
		if len(self._id) == 0: self._id = doc_id

		if len(self.save_dir) > 0:
			with open(os.path.join(self.save_dir, doc_id), 'w') as file:
				json.dump(record, file)


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


	'''
	Remove this Record from database

	outputs:
	count (bool) True if deleted, else False
	'''
	def remove(self):
		res = self.col.delete_one({ '_id': ObjectId(self._id) })
		return bool(res.deleted_count)

	