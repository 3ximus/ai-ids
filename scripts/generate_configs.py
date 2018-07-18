import configparser, os


BASEDIR = '../configs/')

conf = configparser.ConfigParser(allow_no_value=True)
conf.optionxform=str
conf.read(BASEDIR + 'ids.cfg') # base configuration


classifiers = [
	['sklearn.neighbors', '%(classifier-module)s.KNeighborsClassifier(n_neighbors=5)'], # KNN
	['sklearn.neural_network', '%(classifier-module)s.MLPClassifier(activation=\'logistic\', solver=\'lbfgs\', alpha=1e-5, hidden_layer_sizes=(50,30), random_state=1)'], # MLP
	['sklearn.ensemble', '%(classifier-module)s.RandomForestRegressor(max_depth=15, random_state=0)'], # Random Forest
	# Regressors
	['sklearn.tree', '%(classifier-module)s.DecisionTreeClassifier(criterion=\'entropy\', max_depth=10)', 'regressor'], # Decision Tree
	['sklearn.linear_model', '%(classifier-module)s.LogisticRegression()', 'regressor'], # Logistic Regression
	['sklearn.svm', '%(classifier-module)s.SVC(kernel=\'rbf\', C=1)', 'regressor']
]

sections = ['l1', 'l2-fastdos', 'l2-portscan', 'l2-bruteforce']


unique = 0
def get_unique():
	global unique
	unique += 1
	return unique

def select_classsifier(section):
	if section == 4:
		with open(BASEDIR + 'test/%d.cfg' % get_unique(), 'w') as fp:
			conf.write(fp)
		return
	for cls in classifiers[:3]: # skip regressors for now
		conf[sections[section]]['classifier-module'] = cls[0]
		conf[sections[section]]['classifier'] = cls[1]
		if len(cls) == 3:
			conf[sections[section]]['regressor'] = ''
		elif conf.has_option(sections[section], 'regressor'): # ensure no regressor
			conf.remove_option(sections[section], 'regressor')
		select_classsifier(section+1)


# RUN

if not os.path.isdir(BASEDIR + 'test'): os.mkdir(BASEDIR + 'test')
select_classsifier(0)
