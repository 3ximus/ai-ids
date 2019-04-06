"""This file contains the class NodeModel

AUTHORS:

Joao Meira <joao.meira@tekever.com>
Fabio Almeida <fabio4335@gmail.com>
"""

import os, pickle, hashlib
from lib.log import Stats, Logger
import numpy as np

class NodeModel:
	'''Class to train and apply classifier/regressor models'''

	def __init__(self, node_name, config, verbose=False):
		'''Create a model object with given node_name, configuration and labels gathered from label_section in config options'''

		self.verbose = verbose
		self.node_name = node_name

		# get options
		self.attack_keys   = config.options(config.get(node_name, 'labels'))
		n_labels		   = len(self.attack_keys)
		outputs			   = [[1 if j == i else 0 for j in range(n_labels)] for i in range(n_labels)]

		self.outputs	   = dict(zip(self.attack_keys, outputs))
		self.force_train   = config.has_option(node_name, 'force_train')
		self.use_regressor = config.has_option(node_name, 'regressor')
		self.unsupervised  = config.has_option(node_name, 'unsupervised')
		self.save_path	   = config.get(node_name, 'saved-model-path')

		if self.use_regressor or self.unsupervised: # generate outputs for unsupervised or regressor models
			self.outputs = {k:np.argmax(self.outputs[k]) for k in self.outputs}

		self.label_map	   = dict(config.items(config.get(node_name, 'labels-map'))) if config.has_option(node_name, 'labels-map') else dict()

		# model settings
		self.classifier				  = config.get(node_name, 'classifier')
		self.classifier_module		  = config.get(node_name, 'classifier-module') if config.has_option(node_name, 'classifier-module') else None

		self.feature_selection		  = config.get(node_name, 'feature-selection') if config.has_option(node_name, 'feature-selection') else None
		self.feature_selection_module = config.get(node_name, 'feature-selection-module') if config.has_option(node_name, 'feature-selection-module') else None

		self.scaler					  = config.get(node_name, 'scaler') if config.has_option(node_name, 'scaler') else None
		self.scaler_module			  = config.get(node_name, 'scaler-module') if config.has_option(node_name, 'scaler-module') else None


		self.model = None # leave uninitialized (run self.train)
		self.saved_model_file = None
		self.saved_feature_selection_file = None
		self.saved_scaler_file = None
		self.stats = Stats(self)
		self.logger = Logger(config.get('ids', 'log-dir'), node_name, self.classifier.split('\n')[0].strip('()').split('.')[-1])


	@staticmethod
	def save_model(filename, clfmodel):
		'''Save the model to disk'''
		if not os.path.isdir(os.path.dirname(filename)) and os.path.dirname(filename) != '':
			os.makedirs(os.path.dirname(filename))
		model_file = open(filename,'wb')
		pickle.dump(clfmodel, model_file)
		model_file.close()

	@staticmethod
	def load_model(filename):
		'''Load the model from disk'''
		if not os.path.isfile(filename):
			print("File %s does not exist." % filename)
			exit()
		model_file = open(filename, 'rb')
		loaded_model = pickle.load(model_file)
		model_file.close()
		return loaded_model

	@staticmethod
	def gen_saved_model_pathname(base_path, train_filename, classifier_settings):
		'''Generate name of saved model file

			Parameters
			----------
			- base_path				   base path name
			- train_filename		   train file name
			- classifier_settings	   string for with classifier training settings
		'''
		used_model_md5 = hashlib.md5()
		used_model_md5.update(classifier_settings.encode('utf-8'))
		return base_path + '/%s-%s' % (train_filename[:-4].replace('/','-'), used_model_md5.hexdigest()[:7])

	# normal csvs
	def parse_csvdataset(self, fd):
		'''Parse entire dataset and return processed np.array with x and y'''
		flow_ids, x_in, y_in = [], [], []
		feature_string = fd.readline().split(',')
		index_subset=[]
		for elem in feature_string:
			if 'iat' not in elem and 'sec' not in elem and 'duration' not in elem and elem != 'label\n' and elem !='flow_id':
				index_subset.append(feature_string.index(elem))
		for line in fd:
			tmp = line.strip('\n').split(',')
			#x_in.append(tmp[1:-1])
			x_in.append([tmp[i] for i in index_subset])
			y_in.append(tmp[-1]) # choose result based on label
			flow_ids.append(tmp[0])
		return self.process_data(x_in, y_in, flow_ids)

	def yield_csvdataset(self, fd, n_chunks):
		'''Iterate over data, yielding np.array with x and y in chunks of size n_chunks'''
		flow_ids, x_in, y_in = [], [], []
		feature_string = fd.readline()
		feature_string = feature_string.split(',')
		index_subset=[]
		for elem in feature_string:
			if 'iat' not in elem and 'sec' not in elem and 'duration' not in elem and elem != 'label\n' and elem !='flow_id':
				index_subset.append(feature_string.index(elem))
		for i, line in enumerate(fd):
			tmp = line.strip('\n').split(',')
			#x_in.append(tmp[1:-1])
			x_in.append([tmp[j] for j in index_subset])
			y_in.append(tmp[-1]) # choose result based on label
			flow_ids.append(tmp[0])
			if (i+1) % n_chunks == 0:
				yield self.process_data(x_in, y_in, flow_ids)
				x_in, y_in, flow_ids = [], [], []
		yield self.process_data(x_in, y_in, flow_ids)

	def process_data(self, x, labels, flow_ids):
		'''Process data, y must be a list of labels, returns list with both lists converted to np.array'''
		y = []
		for label in labels:
			# try to apply label to elf.outputs, if not existent use label mapping to find valid label conversion
			if label in self.outputs:
				y.append(self.outputs[label]) # encode label into categorical ouptut classes
			elif label in self.label_map:
				y.append(self.outputs[self.label_map[label]]) # if an error ocurrs here your label conversion is wrong
			else:
				self.logger.log("%s : Unknown label %s. Add it to correct mapping section in config file" % (self.node_name, label), self.logger.error, self.verbose)
				exit()
		x = np.array(x, dtype='float64')
		y = np.array(y, dtype='int8')
		flow_ids = np.array(flow_ids)
		return [x, y, labels, flow_ids]

	def train(self, train_filename, disable_load=False):
		'''Create or load train model from given dataset and apply it to the test dataset

			If there is already a created model with the same classifier and train dataset
				it will be loaded, otherwise a new one is created and saved

			Parameters
			----------
			- train_filename	  filename of the train dataset
			- disable_load		  disable load of trained classifier models
		'''
		# generate model filename
		self.saved_model_file = self.gen_saved_model_pathname(self.save_path, train_filename, self.classifier)
		self.saved_scaler_file = (os.path.dirname(self.saved_model_file) + '/scalerX_' + self.node_name) if self.scaler else None
		self.saved_feature_selection_file = (os.path.dirname(self.saved_model_file) + '/FeatureSelection_' + self.node_name) if self.feature_selection else None

		if os.path.isfile(self.saved_model_file) and not disable_load and not self.force_train:
			# LOAD MODEL
			self.logger.log("%s importing model from %s" % (self.node_name, self.saved_model_file),self.logger.normal, self.verbose)
			self.model = self.load_model(self.saved_model_file)
		else:
			# CREATE NEW MODEL
			with open(train_filename, 'r') as fd:
				X_train, y_train, _, _ = self.parse_csvdataset(fd)

			# scaler setup
			if self.scaler_module:
				exec('import '+ self.scaler_module) # import scaler module
			if self.scaler:
				scaler = eval(self.scaler).fit(X_train)
				X_train = scaler.transform(X_train)    # normalize
				self.save_model(self.saved_scaler_file, scaler)

			# feature selection
			if self.feature_selection_module:
				exec('import '+ self.feature_selection_module) # import feature selection module
			if self.feature_selection:
				fs_model = eval(self.feature_selection).fit(X_train)
				X_train = fs_model.transform(X_train) # apply dimension reduction
				self.save_model(self.saved_feature_selection_file, fs_model)

			# classifier setup
			if self.classifier_module:
				exec('import '+ self.classifier_module) # import classifier module
			self.model = eval(self.classifier)

			# train and save the model
			self.logger.log("%s : Training" %  self.node_name, self.logger.normal, True)
			try:
				if self.unsupervised:
					self.model.fit(X_train)
				else:
					self.model.fit(X_train, y_train)
			except ValueError as err:
				self.logger.log("%s : Problem found when training model, this classifier might be a regressor:\n%s\nIf it is, use 'regressor' option in configuration file" % (self.node_name, self.model), self.logger.error)
				self.logger.log("%s : %s" % (self.node_name, err), self.logger.error, True)
				exit()
			except TypeError:
				self.logger.log("%s : Problem found when training model, this classifier might not be unsupervised:\n%s" % (self.node_name, self.model), self.logger.error)
				exit()
			self.save_model(self.saved_model_file, self.model)
		self.logger.log("%s model: %s" % (self.node_name, self.classifier), self.logger.normal, True)
		return self.model

	def predict(self, test_data):
		'''Apply a created model to given test_data (tuple with data input and data labels) and return predicted classification'''

		if not self.model:
			self.logger.log("%s : The model hasn't been trained or loaded yet. Run NodeModel.train" % self.node_name, self.logger.error)
			exit()
		X_test, y_test, _, flow_ids = test_data

		# apply network to the test data
		if self.saved_scaler_file and os.path.isfile(self.saved_scaler_file):
			scaler = self.load_model(self.saved_scaler_file)
			try:
				X_test = scaler.transform(X_test) # normalize
			except ValueError as err:
				self.logger.log("%s : Transforming with scaler. %s" % (self.node_name, err), self.logger.error)
				exit()

		# apply feature selection transformation to test data
		if self.saved_feature_selection_file and os.path.isfile(self.saved_feature_selection_file):
			fs_model = self.load_model(self.saved_feature_selection_file)
			try:
				X_test = fs_model.transform(X_test) # dimension reduction
			except ValueError as err:
				self.logger.log("%s : Performing feature selection. %s" % (self.node_name, err), self.logger.error)
				exit()

		self.logger.log("%s : Predicting on #%d samples" % (self.node_name, len(X_test)), self.logger.normal, self.verbose)
		try:
			y_predicted = self.model.predict(X_test)
		except ValueError as err:
			self.logger.log("%s : Predicting. %s" % (self.node_name, err), self.logger.error)
			exit()

		if not self.use_regressor and not self.unsupervised:
			y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)

		if self.unsupervised:
			y_predicted[y_predicted == 1] = 0
			y_predicted[y_predicted == -1] = 1
		self.stats.update(y_predicted, y_test)
		return y_predicted,flow_ids


