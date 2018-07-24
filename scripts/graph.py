# import plotly, numpy
# import plotly.graph_objs as go
import re, glob

L1_RE = re.compile(r'.*l1 model: sklearn\..+\.(.*)')
L2_DOS_RE = re.compile(r'.*l2\-fastdos model: sklearn\..+\.(.*)')
L2_PS_RE = re.compile(r'.*l2\-portscan model: sklearn\..+\.(.*)')
L2_BF_RE = re.compile(r'.*l2\-bruteforce model: sklearn\..+\.(.*)')

ACC = re.compile(r'(Overall Acc) = .+m([\.0-9]+)')
REC = re.compile(r'(Recall) = (.+)')
MISS = re.compile(r'(Miss Rate) = (.+)')
SPEC = re.compile(r'(Specificity) = (.+)')
FALL = re.compile(r'(Fallout) = (.+)')
PREC = re.compile(r'(Precision) = (.+)')
FSCORE = re.compile(r'(F1 score) = (.+)')
MCC = re.compile(r'(Mcc) = (.+)')

def parse_l1(fd, models):
	for line in fd:
		match = re.search(L1_RE, line)
		if match:
			if match.group(1) not in models:
				models[match.group(1)] = []
				return match.group(1)
			return match.group(1)
	raise ValueError('Didn\'t find layer 1 in %s' % fd.name)

def get_stats(fd):
	stats = {}
	stat_read = False
	no_stat_read = False
	for line in fd:
		for patt in (ACC, REC, MISS, SPEC, FALL, PREC, FSCORE, MCC):
			match = re.search(patt, line)
			if match:
				stats[match.group(1)] = float(match.group(2))
				no_stat_read = False
				stat_read = True
				break
			else: no_stat_read = True
		if no_stat_read and stat_read:
			break
	return stats

def parse_l2(fd, models, mod):
	algos = []
	for line in fd:
		match = re.search(L2_DOS_RE, line)
		if match:
			algos.append((match.group(1), get_stats(fd)))
			continue
		match = re.search(L2_BF_RE, line)
		if match:
			algos.append((match.group(1), get_stats(fd)))
			continue
		match = re.search(L2_PS_RE, line)
		if match:
			algos.append((match.group(1), get_stats(fd)))
	assert(len(algos) == 3)
	models[mod].append(algos)

models = {}

# { L1 Model :
#       [   # description of files with that L1 Model
#			[ (L2 Dos, {stats}) (L2 PS, {stats}) (L2 BR, {stats}) ],
#			[ ... ]
# 		]
# }

for log_file in glob.glob('log/*.log'):
	with open(log_file, 'r') as fd:
		mod = parse_l1(fd, models)
		fd.seek(0)
		parse_l2(fd, models, mod)

# x_data = ['Carmelo Anthony', 'Dwyane Wade',
# 			'Deron Williams', 'Brook Lopez',
# 			'Damian Lillard', 'David West',]

# y0 = numpy.random.randn(50)-1
# y1 = numpy.random.randn(50)+1
# y2 = numpy.random.randn(50)
# y3 = numpy.random.randn(50)+2
# y4 = numpy.random.randn(50)-2
# y5 = numpy.random.randn(50)+3

# y_data = [y0,y1,y2,y3,y4,y5]

# colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

# traces = []

# for xd, yd, cls in zip(x_data, y_data, colors):
# 	traces.append(go.Box(
# 		y=yd,
# 		name=xd,
# 		boxpoints='all',
# 		jitter=0.5,
# 		whiskerwidth=0.2,
# 		fillcolor=cls,
# 		marker=dict(
# 			size=2,
# 		),
# 		line=dict(width=1),
# 	))

# layout = go.Layout(
# 	title='Points Scored by the Top 9 Scoring NBA Players in 2012',
# 	yaxis=dict(
# 		autorange=True,
# 		showgrid=True,
# 		zeroline=True,
# 		dtick=5,
# 		gridcolor='rgb(255, 255, 255)',
# 		gridwidth=1,
# 		zerolinecolor='rgb(255, 255, 255)',
# 		zerolinewidth=2,
# 	),
# 		margin=dict(
# 		l=40,
# 		r=30,
# 		b=80,
# 		t=100,
# 	),
# 		paper_bgcolor='rgb(243, 243, 243)',
# 		plot_bgcolor='rgb(243, 243, 243)',
# 		showlegend=False
# 	)

# fig = go.Figure(data=traces, layout=layout)
# plotly.offline.plot(fig)

