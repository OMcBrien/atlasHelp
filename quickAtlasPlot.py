#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import atlasHelp as ah
import sys
import getopt

usage = """
Usage quickAtlasPlot.py -h -f <file_name> -t <threshold>

Options:
-h, show this help message
-f <file_name>, specify the file to plot
-t <threshold>, specify the cut-off threshold for SNR [default is 5]
"""

argv = sys.argv[1:]

try:
	opts, args = getopt.getopt(argv, 'h,f:,t:')
# 	print('opts = ', opts)
# 	print('args = ', args)
except getopt.GetoptError:
	sys.exit('\nIncorrect usage.\nPlease run "quickAtlasHelp.py -h" for help.\n')

for o, a in opts:
	if o == '-h':
		print(usage)
		sys.exit()
	if o == '-f':
		print(a)
		name = a
	if o == '-t':
		print(a)
		snr_threshold = float(a)

data = ah.photometry_prep(name)
data = ah.snr_cut(data, snr_threshold)

fig = ah.atlas_plot(data)

plt.title(name.strip('.csv'))

plt.tight_layout()
plt.show(fig)