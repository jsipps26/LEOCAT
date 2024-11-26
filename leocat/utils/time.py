

import numpy as np

def date_to_ymd(date):
	# change date1 into y,m,d to input into date_to_jd
	date_ymd = []
	for i in range(6): # y,m,d,h,m,s
		val = None
		if i < len(date):
			val = date[i]
		if i < 3:
			if val is None:
				val = 1
			date_ymd.append(val)
		elif i == 3:
			# hour
			if val is None:
				val = 0
			date_ymd[-1] = date_ymd[-1] + val/24
		elif i == 4:
			# minute
			if val is None:
				val = 0
			date_ymd[-1] = date_ymd[-1] + val/(24*60)
		elif i == 5:
			# second
			if val is None:
				val = 0
			date_ymd[-1] = date_ymd[-1] + val/(24*60*60)

	return date_ymd


def JD_to_date_range(JD1,JD2):
	date1 = list(jd_to_date(JD1))
	date2 = list(jd_to_date(JD2))
	# date1[2] -= 1
	# date2[2] += 1
	date1[2] = np.floor(date1[2])
	date2[2] = np.ceil(date2[2])
	JD_lower = date_to_jd(*date1)
	JD_upper = date_to_jd(*date2)
	num_days = int(np.round(JD_upper-JD_lower))
	JD_days = JD_lower + np.arange(num_days)
	dates = jd_to_date_vec(JD_days)
	dates = dates.astype(int)
	if len(dates.shape) == 1:
		dates = np.array([dates])

	return dates
	


def ymd_to_str(ymd):
	ymd_str = ''
	for i, val in enumerate(ymd):
		ymd_str += '%s' % str(val).zfill(2)
		if i < 2:
			ymd_str += '-'

	return ymd_str


def ymdhms_to_val(ymdhms):
	y = int(ymdhms[0:4])
	m = int(ymdhms[4:6])
	d = int(ymdhms[6:8])
	h = int(ymdhms[8:10])
	min0 = int(ymdhms[10:12])
	s = int(ymdhms[12:14])
	return y, m, d, h, min0, s

def jd_to_ymdhms(jd):
	date = jd_to_date(jd)
	day = date[2]
	hour = (day % 1.0) * 24
	minute = (hour % 1.0) * 60
	sec = (minute % 1.0) * 60

	return int(date[0]), int(date[1]), int(date[2]), \
			int(hour), int(minute), sec


def ymdhms_to_jd(y,m,d,h,min0,s):
	day = d + h/24 + min0/60/24 + s/86400
	JD = date_to_jd(y, m, day)
	return JD

def JD_to_datetime(JD, rounding=np.floor):
	import datetime
	y, m, d, h, min0, s = jd_to_ymdhms(JD)
	s_input = int(rounding(s))
	# print(s_input)
	if s_input == 60:
		# b/c datetime sec must be [0,...,59]
		date = datetime.datetime(y, m, d, h, min0, s_input-1)
		date = date + datetime.timedelta(seconds=1)
	else:
		date = datetime.datetime(y, m, d, h, min0, s_input)

	# date = datetime.datetime(y, m, d, h, min0, int(rounding(s)))
	return date

	
def jd_to_date(jd):
	# https://gist.github.com/jiffyclub/1294443

	import math
	"""
	Convert Julian Day to date.
	
	Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
		4th ed., Duffet-Smith and Zwart, 2011.
	
	Parameters
	----------
	jd : float
		Julian Day
		
	Returns
	-------
	year : int
		Year as integer. Years preceding 1 A.D. should be 0 or negative.
		The year before 1 A.D. is 0, 10 B.C. is year -9.
		
	month : int
		Month as integer, Jan = 1, Feb. = 2, etc.
	
	day : float
		Day, may contain fractional part.
		
	Examples
	--------
	Convert Julian Day 2446113.75 to year, month, and day.
	
	>>> jd_to_date(2446113.75)
	(1985, 2, 17.25)
	
	"""
	jd = jd + 0.5
	
	F, I = math.modf(jd)
	I = int(I)
	
	A = math.trunc((I - 1867216.25)/36524.25)
	
	if I > 2299160:
		B = I + 1 + A - math.trunc(A / 4.)
	else:
		B = I
		
	C = B + 1524
	
	D = math.trunc((C - 122.1) / 365.25)
	
	E = math.trunc(365.25 * D)
	
	G = math.trunc((C - E) / 30.6001)
	
	day = C - E + F - math.trunc(30.6001 * G)
	
	if G < 13.5:
		month = G - 1
	else:
		month = G - 13
		
	if month > 2.5:
		year = D - 4716
	else:
		year = D - 4715
		
	return year, month, day


def date_to_jd(year,month,day):
	# https://gist.github.com/jiffyclub/1294443

	import math
	"""
	Convert a date to Julian Day.
	
	Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
		4th ed., Duffet-Smith and Zwart, 2011.
	
	Parameters
	----------
	year : int
		Year as integer. Years preceding 1 A.D. should be 0 or negative.
		The year before 1 A.D. is 0, 10 B.C. is year -9.
		
	month : int
		Month as integer, Jan = 1, Feb. = 2, etc.
	
	day : float
		Day, may contain fractional part.
	
	Returns
	-------
	jd : float
		Julian Day
		
	Examples
	--------
	Convert 6 a.m., February 17, 1985 to Julian Day
	
	>>> date_to_jd(1985,2,17.25)
	2446113.75
	
	"""
	if month == 1 or month == 2:
		yearp = year - 1
		monthp = month + 12
	else:
		yearp = year
		monthp = month
	
	# this checks where we are in relation to October 15, 1582, the beginning
	# of the Gregorian calendar.
	if ((year < 1582) or
		(year == 1582 and month < 10) or
		(year == 1582 and month == 10 and day < 15)):
		# before start of Gregorian calendar
		B = 0
	else:
		# after start of Gregorian calendar
		A = math.trunc(yearp / 100.)
		B = 2 - A + math.trunc(A / 4.)
		
	if yearp < 0:
		C = math.trunc((365.25 * yearp) - 0.75)
	else:
		C = math.trunc(365.25 * yearp)
		
	D = math.trunc(30.6001 * (monthp + 1))
	
	jd = B + C + D + day + 1720994.5
	
	return jd


def date_to_jd_vec(ymd):

	year, month, day = ymd.T

	yearp = year
	monthp = month
	b = (month == 1) | (month == 2)
	yearp[b] = year[b] - 1
	monthp[b] = month[b] + 12

	A = np.trunc(yearp / 100.0).astype(int)
	B = 2 - A + np.trunc(A / 4.0).astype(int)
	b = (year < 1582) | \
		((year == 1582) & (month < 10)) | \
		((year == 1582) & (month == 10) & (day < 15))
	#
	B[b] = 0

	C = np.trunc(365.25 * yearp).astype(int)
	b = yearp < 0
	C[b] = np.trunc((365.25 * yearp[b]) - 0.75).astype(int)

	D = np.trunc(30.6001 * (monthp + 1)).astype(int)
	jd = B + C + D + day + 1720994.5

	return jd




def jd_to_date_vec(jd):
	# import math
	"""
	Convert Julian Day to date.
	
	Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
		4th ed., Duffet-Smith and Zwart, 2011.
	
	Parameters
	----------
	jd : float
		Julian Day
		
	Returns
	-------
	year : int
		Year as integer. Years preceding 1 A.D. should be 0 or negative.
		The year before 1 A.D. is 0, 10 B.C. is year -9.
		
	month : int
		Month as integer, Jan = 1, Feb. = 2, etc.
	
	day : float
		Day, may contain fractional part.
		
	Examples
	--------
	Convert Julian Day 2446113.75 to year, month, and day.
	
	>>> jd_to_date(2446113.75)
	(1985, 2, 17.25)
	
	"""

	n = len(jd)
	if not (type(jd) == np.ndarray):
		jd = np.array([jd])
		n = 1

	jd = jd + 0.5
	
	F, I = np.modf(jd)
	I = I.astype(int)
	
	A = np.trunc((I - 1867216.25)/36524.25)

	B = I
	B[I > 2299160] = I + 1 + A - np.trunc(A / 4.)
	
	# if I > 2299160:
	# 	B = I + 1 + A - np.trunc(A / 4.)
	# else:
	# 	B = I
		
	C = B + 1524
	
	D = np.trunc((C - 122.1) / 365.25)
	
	E = np.trunc(365.25 * D)
	
	G = np.trunc((C - E) / 30.6001)
	
	day = C - E + F - np.trunc(30.6001 * G)
	
	month = G - 13
	bG = G < 13.5
	if bG.sum() > 0:
		month[bG] = G[bG] - 1

	# if G < 13.5:
	# 	month = G - 1
	# else:
	# 	month = G - 13

	year = D - 4715
	bM = month > 2.5
	# print(month, G, bG)
	if bM.sum() > 0:
		year[bM] = D[bM] - 4716
		
	# if month > 2.5:
	# 	year = D - 4716
	# else:
	# 	year = D - 4715
		
	if n == 1:
		return np.transpose([year, month, day])[0]
	else:
		return np.transpose([year, month, day])
		