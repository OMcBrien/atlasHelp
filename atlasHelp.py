import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def ndt_stripper(ndt):
	
"""
Returns array of non-detections from array of detections and non-detections mixed

	Parameters:
	ndt (array-like): Array of non-detection and detection data combined

	Returns:
	ndt (array-like): Original array with non-detections cast as np.float64 and detections replaced with np.nan

"""
	
	for i in range(0, len(ndt)):
		if ndt[i][0] == '>':
			ndt[i] = float(ndt[i][1:len(ndt[i])])
		else:
			ndt[i] = np.nan

	ndt = ndt.astype(np.float)
	
	return ndt

# =====

def photometry_prep(obj_name):

"""
Reads .csv file containing ATLAS data, cleans and returns relevant columns as a 2D array

	Parameters:
	obj_name (str): path to .csv file containing ATLAS data

	Returns:
	data (np.ndarray): cleaned Pandas dataframe containing mjd, detection magnitude, non-detection limit, snr and filter data
"""
	
	data = pd.read_csv(obj_name, usecols = ['mjd', 'mag', 'magerr', 'snr', 'filter'])
	
	mag = np.array(data['mag'])
	ndt = np.array(data['mag'])
	err = np.array(data['magerr'])
	snr = np.array(data['snr'])
	
	for i in range(0, len(mag)):
	
		if not ndt[i]:
			ndt[i] = np.nan
			continue
		elif pd.isnull(ndt[i]): 
			ndt[i] = np.nan
			continue
		elif ndt[i] == 'None':
			ndt[i] = np.nan
			mag[i] = np.nan
			err[i] = np.nan
			snr[i] = np.nan
			continue
		
		else:
			pass
	
	
		if str(ndt[i][0]) == '>':
			ndt[i] = float(ndt[i][1:len(ndt[i])])
			mag[i] = np.nan
			err[i] = np.nan
			snr[i] = 0.0
	
		else:
			ndt[i] = np.nan
	
	data['mag'] = pd.Series(mag, index = data.index)
	data['ndt'] = pd.Series(ndt, index = data.index)
	data['magerr'] = pd.Series(err, index = data.index)
	data['snr'] = pd.Series(snr, index = data.index)
	
	return data
	
# =====

def atlas_plot(data):

"""
Plots ATLAS data made using atlasHelp.photometry_prep()

	Parameters:
	data (pd.DataFrame): data frame object produced by atlasHelp.photometry_prep()

	Returns:
	fig (matplotlib figure): produced figure
"""

	SMALL_SIZE = 14
	MEDIUM_SIZE = 18
	BIGGER_SIZE = 25

	plt.rc('font', size=SMALL_SIZE)          	# controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     	# fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    	# fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    	# fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    	# fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE - 1)    	# legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  	# fontsize of the figure title
	
	plt.rcParams["font.family"] = "serif"
	plt.rcParams['mathtext.fontset'] = 'dejavuserif'

	
	marker_edge_width = 0.75
	marker_edge_colour = 'k'
	marker_size = 8
	error_colour = 'k'
	cap_size = 5

	fig = plt.figure(figsize=(10,8))
	plt.gca().invert_yaxis()
	
	mjd = np.array(data['mjd']).astype(float)
	mag = np.array(data['mag']).astype(float)
	ndt = np.array(data['ndt']).astype(float)
	err = np.array(data['magerr']).astype(float)
	snr = np.array(data['snr']).astype(float)
	filter = np.array(data['filter']).astype(str)
	
	
	
	cyan=np.where(filter == 'c')
	orange=np.where(filter == 'o')
	mjd_cyan=mjd[cyan]
	mag_cyan=mag[cyan]
	ndt_cyan=ndt[cyan]
	err_cyan=err[cyan]
	mjd_orange=mjd[orange]
	mag_orange=mag[orange]
	ndt_orange=ndt[orange]
	err_orange=err[orange]

	with np.errstate(invalid = 'ignore'):
		pos_cyan = np.where(mag_cyan > 0.0)
		neg_cyan = np.where(mag_cyan < 0.0)
		pos_orange = np.where(mag_orange > 0.0)
		neg_orange = np.where(mag_orange < 0.0)
	
	plt.errorbar(mjd_cyan[pos_cyan],mag_cyan[pos_cyan],err_cyan[pos_cyan],capsize=cap_size,linestyle='None',marker='o',color='#00ffff',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size,capthick=marker_edge_width*1.5,ecolor=error_colour)
	plt.plot(mjd_cyan,ndt_cyan,linestyle='None',marker='v',color='#00ffff',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size-1)
	plt.plot([],[],linestyle='None',marker='o',color='#00ffff',label='ATLAS cyan',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size)
	
	plt.errorbar(mjd_orange[pos_orange],mag_orange[pos_orange],err_orange[pos_orange],capsize=cap_size,linestyle='None',marker='o',color='#f97306',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size,capthick=marker_edge_width*1.5,ecolor=error_colour)
	plt.plot(mjd_orange,ndt_orange,linestyle='None',marker='v',color='#f97306',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size-1)
	plt.plot([],[],linestyle='None',marker='o',color='#f97306',label='ATLAS orange',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size)
	
	if len(mag_cyan[neg_cyan]) or len(mag_orange[neg_orange]) > 0:
		plt.errorbar(mjd_cyan[neg_cyan],abs(mag_cyan[neg_cyan]),err_cyan[neg_cyan],capsize=cap_size,linestyle='None',marker='D',color='#00ffff',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size,capthick=marker_edge_width*1.5,ecolor=error_colour)
		plt.plot([],[],linestyle='None',marker='D',color='#00ffff',label='ATLAS cyan',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size)
	
		plt.errorbar(mjd_orange[neg_orange],abs(mag_orange[neg_orange]),err_orange[neg_orange],capsize=cap_size,linestyle='None',marker='D',color='#f97306',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size,capthick=marker_edge_width*1.5,ecolor=error_colour)
		plt.plot([],[],linestyle='None',marker='D',color='#f97306',label='ATLAS orange',mew=marker_edge_width,mec=marker_edge_colour,ms=marker_size)
	
	plt.ylabel('Apparent magnitude')
	plt.xlabel('MJD')
	plt.legend(fontsize='medium',numpoints=1,frameon=False)
	
	return fig

# ========================================================================================

def snr_cut(data, snr_threshold = 5.0):

"""
Imposes a signal-to-noise cut instead of the default 3 sigma cut on detections

	Parameters:
	data (pd.DataFrame): data frame object produced by atlasHelp.photometry_prep()
	snr_threshold (float): signal-to-noise level at which the cut is performed

	Returns:
	data (pd.DataFrame): data frame object produced by atlasHelp.photometry_prep()
"""

	mjd = np.array(data['mjd']).astype(float)
	mag = np.array(data['mag']).astype(float)
	ndt = np.array(data['ndt']).astype(float)
	err = np.array(data['magerr']).astype(float)
	snr = np.array(data['snr']).astype(float)
	filter = np.array(data['filter']).astype(str)

# 	print('Imposing SNR cut at %g.\n' %snr_threshold)

	for i in range(0,len(mjd)):
		if ( ~np.isfinite( mag[i])):
			pass
			

		elif (snr[i] >= snr_threshold) and (np.isfinite( mag[i])):
			pass

		elif (snr[i] < snr_threshold):# and (~np.isfinite( mag[i])):
			ndt[i] = mag[i]
			mag[i] = np.nan
			err[i] = np.nan

    print('Je suis here!')

	data['mag'] = pd.Series(mjd, index = data.index)
	data['mag'] = pd.Series(mag, index = data.index)
	data['ndt'] = pd.Series(ndt, index = data.index)
	data['magerr'] = pd.Series(err, index = data.index)
	data['snr'] = pd.Series(snr, index = data.index)
	data['filter'] = pd.Series(filter, index = data.index)	

	return data

# ========================================================================================

def night_average(datasq):
	
	sys.exit('#####\nCode not tested. Please do not use just yet.\n#####)

	return None

# 	mjd = np.array(datasq['mjd'])
# 	mag = np.array(datasq['mag'])
# 	ndt = np.array(datasq['ndt']).astype(float)
# 	err = np.array(datasq['magerr'])
# 	snr = np.array(datasq['snr'])
# 	filter = np.array(datasq['filter'])
# 
# 	round_mjd = np.floor( list( mjd))
# 	unique_mjd = np.array( list( dict.fromkeys(round_mjd)))
# 
# 	avg_mjd = np.zeros(len(unique_mjd))
# 	avg_mag = np.zeros(len(unique_mjd))
# 	avg_err = np.zeros(len(unique_mjd))
# 	stddev = np.zeros(len(unique_mjd))
# 	avg_filter = np.empty_like(avg_mjd, dtype=str)
# 
# 	for i in range(0,len(unique_mjd)):
# 
# 		ind=np.where( np.equal( round_mjd, unique_mjd[i]) )
# 		dummy_mjd = mjd[ind]
# 		dummy_mag = mag[ind]
# 		dummy_err = err[ind]
# 		dummy_filter = filter[ind]
# 	
# 		if np.any(dummy_mag < 0.0) and not np.all(dummy_mag < 0.0):
# 			dummy_mag = dummy_mag[np.where( dummy_mag > 0.0)]
# 		
# 		avg_mjd[i] = np.median(dummy_mjd)
# 		avg_mag[i] = np.mean(dummy_mag)
# 		avg_filter[i] = dummy_filter[0]
# 		
# 		if len(dummy_mag) == 1:
# 		
# 			stddev[i] = dummy_err
# 			
# 		else:
# 		
# 			stddev[i] = np.std(dummy_mag)
# 
# # =============================
# 
# 	zeropoints = [21.06916566, 21.60144232]
# 
# 	avg_ndt = np.full(len(unique_mjd), np.nan)
# 	stddev_ndt = np.zeros(len(unique_mjd))
# 
# 	for i in range(0,len(unique_mjd)):
# 		
# 		ind = np.where( np.logical_and( np.equal( round_mjd, unique_mjd[i]), np.isfinite(ndt) ) )
# 		dummy_ndt = np.array(ndt[ind], dtype=float)
# # 		dummy_mag = mag[ind]
# 		dummy_filter = filter[ind]
# 
# 		dummy_ndt = dummy_ndt[ np.isfinite(dummy_ndt) ]
# 		
# # 		print(unique_mjd[i])
# # 		print(avg_mjd[i])
# # 		print(dummy_ndt)
# # 		print(dummy_filter)
# # 		print('\n')
# 		
# # 		if unique_mjd[i] == 57653.0:
# 		
# 		if np.all(dummy_filter) == 'c':
# 			
# 			dummy_flux = 10**(-(dummy_ndt + 48.6 + zeropoints[0])/2.5)
# 			avg_flux = np.mean( dummy_flux)
# 			avg_ndt[i] = -2.5*np.log10( avg_flux) - 48.6 - zeropoints[0]
# # 			stddev_ndt[i] = -2.5*np.log10( np.std(avg_flux)) - 48.6 - zeropoints[0] 
# 
# 
# 		elif np.all(dummy_filter) == 'o':
# 			
# 			dummy_flux = 10**(-(dummy_ndt + 48.6 + zeropoints[1])/2.5)
# 			avg_flux = np.mean( dummy_flux)
# 			avg_ndt[i] = -2.5*np.log10( avg_flux) - 48.6 - zeropoints[1]
# # 			stddev_ndt[i] = -2.5*np.log10( np.std(avg_flux)) - 48.6 - zeropoints[1] 
# 
# 
# 	snr = np.full_like(avg_mjd, 5.0)
# 	
# # 	avg_mjd, avg_mag, avg_ndt, stddev, snr, avg_filter = zip( *sorted( zip( avg_mjd, avg_mag, avg_ndt, stddev, snr, avg_filter)))
# 
# 	datasq = pd.DataFrame({'mjd': avg_mjd, 'mag': avg_mag, 'ndt': avg_ndt, 'magerr': stddev, 'snr': snr, 'filter': avg_filter})
# 
# # 	datasq['mjd'] = pd.Series(avg_mjd, index = datasq.index)
# # 	datasq['mag'] = pd.Series(avg_mag, index = datasq.index)
# # 	datasq['ndt'] = pd.Series(avg_ndt, index = datasq.index)
# # 	datasq['magerr'] = pd.Series(stddev, index = datasq.index)
# # 	datasq['snr'] = pd.Series(snr, index = datasq.index)
# # 	datasq['filter'] = pd.Series(avg_filter, index = datasq.index)
# 
# 	return datasq





