#!/usr/bin/env python
import plotly
from plotly.graph_objs import Scatter, Line, Marker, Figure, Layout
import re, glob

TESTFILENAME_PATTERN = re.compile(r'RESULTS.+\[([^\]]+)\]')
MB_PATTERN = re.compile(r'm[^0-9]*([0-9\.]+).+ \| .+m[^0-9]*([0-9\.]+)')

def parse_entry(fd, data_type):
    count = next(fd) and next(fd)
    ratio = next(fd)
    match = re.search(MB_PATTERN, ratio)
    return match.group(data_type)

result_files = glob.glob('results/results*')
result_files.sort()
malign_data, benign_data = {}, {}
counter = 0
for i, file in enumerate(result_files):
    with open(file ,'r') as fd:
        for line in fd:
            if line.startswith("MALIGN"):
                data = malign_data
                data_type = 2 # MALIGN (Group in MB_PATTERN)
            elif line.startswith("BENIGN"):
                data = benign_data
                data_type = 1 # BENIGN (Group in MB_PATTERN)
            elif "RESULTS" in line:
                match = re.search(TESTFILENAME_PATTERN, line)
                entry = parse_entry(fd, data_type)
                if match.group(1) in data:
                    data[match.group(1)].append(entry)
                else:
                    data[match.group(1)] = [None] * counter + [entry]
    counter += 1
    for data in malign_data, benign_data: # fill data with None if none was attributed in this cycle
        [data[x].append(None) for x in data if len(data[x]) < counter]

try:
	average = lambda i, data: sum([float(x[i]) if x[i] else 0 for x in data.values()]) / len([1 for x in data.values() if x[i]])
	result_files = [r[8:] + " M: %f B: %f" % (average(i, malign_data), average(i, benign_data)) for i, r in enumerate(result_files)]
except ZeroDivisionError:
	print("NOT ENOUGH DATA")
	exit()

malign_traces, benign_traces = [], []
traces = malign_traces
for data in malign_data, benign_data:
    for key in data:
        traces.append(Scatter(
            x = result_files,
            y = data[key],
            # mode = 'lines+markers',
            name = key))
    traces = benign_traces # switch to benign tracers

updatemenus = list([
	dict(buttons = list([
		dict(label="All", method='restyle', args=['visible',[True]*(len(malign_data)+len(benign_data))]),
		dict(label="Malign", method='restyle', args=['visible',[True]*len(malign_data)+[False]*len(benign_data)]),
		dict(label="Benign", method='restyle', args=['visible',[False]*len(malign_data)+[True]*len(benign_data)]),
	]), direction='left', type = 'buttons', x = 0.05, xanchor = 'left', y = 1, yanchor = 'bottom',
	pad = {'b': 1, 'l': 1}),
	])

fig = Figure(data=malign_traces + benign_traces, layout=Layout( title='IDS results data', hovermode='closest', updatemenus=updatemenus))
plotly.offline.plot(fig, filename='results/graph.html', auto_open=False)

