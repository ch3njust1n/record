'''
Author: Justin Chen
Date: 	2.15.2020
#YangGangForever
'''
import os
import time
import atexit
import logging
from datetime import date
from pymongo import MongoClient

class Record(object):
	'''
	inputs	host (string)	MongoDB host name
			port (int)		MongoDB port number
	'''
	def __init__(self, host='localhost', port=27017, database='experiments', collection='parameters'):
		os.mkdir('logs')

		# Setup logger
		self.log = logging.getLogger()
		handler = logging.FileHandler(filename=date.today().strftime("logs/%S-%M-%H-%d-%m-%Y.log"), mode='a')
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

		# Save before process ends
		atexit.register(self.save)


	'''
	Add experiment parameters to experiment record.

	inputs	key   (string)	Name of parameter
			value (any)		Value of parameter
	'''
	def add(self, key, value):
		if type(key) is not str:
			raise Exception('Record.add() key parameter must be type String.')

		self.record[key] = value


	'''
	Save the record to experiment document

 	output: id (string) Inserted document id
	'''
	def save(self):
		doc_id = self.col.insert_one(self.record).inserted_id
		self.log.info('experiment id: {}'.format(doc_id))
		return doc_id

