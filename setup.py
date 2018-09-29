from cx_Freeze import setup, Executable

buildOptions = dict(
		packages = [
			# main modules
			'os', 're', 'time', 'argparse',
			'curses', 'pickle', 'hashlib',
			'threading', 'configparser', 'requests',
			# instaled modules
			'numpy', 'scipy', 'sklearn',
			'certifi', 'chardet', 'decorator',
			'idna', 'jsonschema', 'nbformat',
			'numpy', 'plotly',
			'pytz', 'scipy', 'six',
			'traitlets', 'urllib3',
			# local modules
			'lib.node'],
		excludes = [],
		optimize = 2,
		include_files = [('classifiers/options.cfg', 'classifiers/options.cfg'), 'saved_neural_networks/'],
		zip_include_packages = '*',
		zip_exclude_packages = ['numpy', 'sklearn'])

build_exe_options={"compressed":True}

base = 'Console'

executables = [
    Executable('ids.py', base=base, targetName = 'aids')
]

setup(name='aids',
      version = '0.1',
      description = 'Alpha build',
      options = dict(build_exe = buildOptions),
      executables = executables)
