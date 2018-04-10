#!/usr/bin/env python3
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

fig = Figure(data=malign_traces + benign_traces, layout=Layout( title='IDS results data', hovermode='closest'))
plotly.offline.plot(fig, filename='results/graph.html', auto_open=True)

