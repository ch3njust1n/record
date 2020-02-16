'''
Author: Justin Chen
Date: 	2.15.2020
#YangGangForever
'''
import os
import sys
import csv
import json
import time
import atexit
import logging
import psutil
import platform
from torch import cuda
from datetime import date
from pymongo import MongoClient

class Record(object):
	'''
	inputs	host (string)	MongoDB host name
			port (int)		MongoDB port number
	'''
	def __init__(self, host='localhost', port=27017, database='experiments', collection='parameters'):
		if not os.path.exists('logs'):
			os.mkdir('logs')

		if not os.path.exists('output'):
			os.mkdir('output')

		# Setup logger
		self.name = date.today().strftime('%S-%M-%H-%d-%m-%Y')
		self.log = logging.getLogger()
		handler = logging.FileHandler(filename='logs/'+self.name+'.log', mode='a')
		formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
		handler.setFormatter(formatter)
		self.log.addHandler(handler)
		self.log.setLevel(logging.DEBUG)
		
		# Start MongoDB
		stream = os.popen('mongod')
		self.log.info('starting mongodb')
		self.log.info(stream.read())

		# Connect to MongoDB
		self.client = MongoClient(host, port)

		if database not in self.client.list_database_names():
			log.info('{} does not exist. Will be created when data is inserted.'.format(database))

		self.database = database
		self.collection = collection
		self.record = {}
		self.db = self.client[database]
		self.col = self.db[collection]

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
		self.log.info('experiment id: {}'.format(doc_id))
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
	'''
	def write_csv(self):
		pass


	'''
	'''
	def write_json(self):
		with open('output/'+self.name+'.json', 'w') as f:
			json.dump(self.record, f)


	'''
	'''
	def get(self, record_id):
		pass


