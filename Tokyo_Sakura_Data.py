# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 13:51:03 2022

@author: annam
"""
# make data table for Tokyo

import pandas as pd
import numpy as np
import os
#%%


def make_dates(year_start, year_end):
    dates=[]
    for year in range(year_start,year_end+1):
        for i in range(1,32):
            date = str(year)+"/"+"3"+"/"+str(i)
            dates.append(date)
        for i in range(1,31):
            date = str(year)+"/"+"4"+"/"+str(i)
            dates.append(date)
        for i in range(1,32):
            date = str(year)+"/"+"5"+"/"+str(i)
            dates.append(date)
    return dates
 
dates_ = make_dates(2010, 2020)   
dates_[-1]        
            

temp_2010=[63 ,	50.6 ,	45,
46, 	42.4 ,	39,
54, 	46.2 ,	39,
50, 	45.6 ,	43,
63, 	52.4 ,	43,
55, 	52.6 ,	50,
50, 	44.0 ,	39,
46, 	42.5 ,	39,
43, 	38.9 ,	34,
59, 	43.0 ,	36,
55, 	48.1 ,	37,
57, 	50.9 ,	41,
68, 	62.1 ,	50,
55, 	50.0 ,	46,
63, 	56.2 ,	46,
73, 	61.4 ,	54,
54, 	49.2 ,	46,
55, 	49.8 ,	45,
54, 	46.9 ,	41,
68, 	59.8 ,	45,
68, 	58.7 ,	45,
55, 	49.0 ,	43,
55, 	51.5 ,	46,
50, 	44.9 ,	39,
46, 	42.2 ,	39,
52, 	47.3 ,	43,
54, 	47.2 ,	41,
48, 	42.4 ,	39,
45, 	40.2 ,	37,
48, 	41.9 ,	34,
59, 	49.7 ,	36,

68 ,	59.8, 1,
68, 	61.0,  1,
55, 	51.7, 1,
52, 	48.1, 1,
54, 	48.9 ,	46,
66, 	57.2 ,	52,
64, 	54.0 ,	46,
54, 	47.9 ,	45,
57, 	50.4 ,	45,
66, 	59.0 ,	54,
68, 	60.8 ,	57,
57, 	48.2 ,	43,
68, 	57.2 ,	45,
64, 	56.9 ,	54,
52, 	45.1 ,	43,
46, 	42.4 ,	39,
52 ,	42.2 ,	36,
61, 	52.1 ,	45,
66, 	58.3 ,	52,
64, 	60.4 ,	57,
73, 	63.0 ,	57,
57, 	48.8 ,	46,
50, 	47.1 ,	46,
59, 	51.5 ,	46,
61, 	54.1 ,	45,
64, 	56.7 ,	48,
57, 	55.0 ,	52,
61, 	57.2 ,	55,
72, 	65.5 ,	57,
66, 	59.1 ,	52,

68, 	60.8, 	54,
70, 	62.3, 	54,
72, 	65.2, 	57,
72, 	67.4, 	64,
77, 	68.1, 	63,
75, 	69.7, 	66,
72, 	67.9, 	64,
73, 	66.5, 	63,
77, 	68.1, 	59,
70, 	65.3, 	61,
66, 	62.4, 	59,
70, 	61.8, 	57,
68, 	61.5, 	54,
63, 	58.6, 	55,
63, 	58.5, 	55,
70, 	62.1, 	54,
77 ,	67.1, 	59,
77, 	70.6, 	63,
73, 	69.0, 	66,
70, 	67.6, 	66,
82, 	73.6, 	66,
79, 	72.2, 	66,
68, 	64.8, 	61,
73, 	64.9, 	61,
81, 	74.0, 	68,
72, 	65.0, 	57,
70, 	62.6, 	55,
70, 	64.6, 	61,
63, 	61.0, 	59,
61, 	58.4, 	55,
70, 	62.5, 	57]

temp_2011=[45, 	42.6 ,	41,
50 ,	44.5, 	36,
45 ,	40.0, 	36,
46 ,	40.5, 	34,
52 ,	44.2, 	34,
59 ,	47.8, 	39,
48 ,	40.4, 	36,
48 ,	43.9, 	37,
54 ,	45.8, 	39,
48 ,	42.4, 	37,
52 ,	44.1, 	37,
54 ,	45.7, 	36,
59 ,	51.8, 	43,
66 ,	58.3, 	46,
59 ,	52.2, 	50,
50 ,	45.3, 	39,
48 ,	41.2, 	36,
48 ,	42.7, 	34,
64 ,	55.3, 	48,
64 ,	57.3, 	50,
64 ,	52.8, 	46,
46 ,	45.1, 	43,
48 ,	43.7, 	39,
46 ,	41.7, 	37,
55 ,	45.8, 	39,
52 ,	46.9, 	41,
52 ,	45.5, 	39,
54 ,	46.6, 	41,
57 ,	50.6, 	41,
61 ,	53.1, 	48,
57 ,	49.6, 	43 ,

59 ,	51.9, 	45,
61 ,	55.1, 	50,
52 ,	46.7, 	46,
55 ,	48.8, 	43,
57 ,	49.2, 	41,
64 ,	54.9, 	43,
68 ,	60.6, 	52,
64 ,	62.6, 	61,
64 ,	61.8, 	57,
61 ,	55.3, 	50,
64 ,	57.4, 	50,
61 ,	51.7, 	46,
66 ,	56.9, 	45,
72 ,	60.7, 	52,
72 ,	65.5, 	59,
72 ,	64.5, 	59,
61 ,	55.7, 	50,
63 ,	58.2, 	54,
66 ,	54.6, 	48,
59 ,	51.1, 	45,
59 ,	54.0, 	48,
64 ,	59.4, 	54,
66 ,	61.9, 	59,
68 ,	62.0, 	55,
68, 	59.3, 	52,
68, 	61.4, 	52,
73, 	67.9, 	63,
75, 	66.7, 	59,
64, 	59.1, 	55,
73, 	63.8, 	55,
70, 	65.3, 	59,
73, 	66.6, 	59,
64, 	61.0, 	57,
70, 	62.0, 	57,
63, 	59.9, 	57,
66, 	59.9, 	55,
64, 	62.6, 	61,
75, 	67.2, 	59,
73, 	67.3, 	59,
77, 	71.3, 	64,
64, 	59.5, 	57,
63, 	60.2, 	57,
73, 	67.0, 	61,
75, 	67.8, 	61,
75, 	68.0, 	61,
73, 	68.0, 	64,
70, 	64.0, 	59,
73, 	65.0, 	59,
75, 	68.4, 	61,
77, 	70.5, 	66,
77, 	71.6, 	68,
82, 	70.3, 	61,
64, 	60.7, 	57,
63, 	58.0, 	55,
73, 	64.7, 	55,
70, 	65.6, 	61,
68, 	65.2, 	63,
68, 	65.1, 	63,
66, 	64.8, 	63,
70, 	64.3, 	63,
63, 	59.8, 	55 
]

temp_2012=[52, 	46.7 ,	39,
50 ,	45.1 ,	41,
50 ,	44.4 ,	39,
45 ,	42.3 ,	41,
50 ,	42.5 ,	37,
54 ,	51.4 ,	48,
55 ,	52.5 ,	52,
52 ,	48.4 ,	45,
48 ,	45.6 ,	43,
45 ,	41.9 ,	37,
43 ,	43.0 ,	43,
52 ,	45.7 ,	37,
50 ,	43.7 ,	39,
50 ,	44.3 ,	39,
55 ,	46.3 ,	37,
52 ,	45.0 ,	39,
52 ,	45.4 ,	41,
54 ,	49.3 ,	45,
52 ,	48.4 ,	45,
52 ,	46.1 ,	39,
52 ,	46.3 ,	41,
55 ,	48.6 ,	39,
54 ,	49.3 ,	46,
52 ,	46.7 ,	45,
54 ,	49.1 ,	41,
52 ,	46.7 ,	41,
57 ,	48.9 ,	41,
59 ,	50.7 ,	43,
61 ,	51.4 ,	45,
66 ,	59.5 ,	54,
66 ,	58.4 ,	48, 
59 ,	50.7 ,	43,
57 ,	50.1 ,	45,
61 ,	57.5 ,	52,
59 ,	52.6 ,	46,
66 ,	58.8 ,	52,
59 ,	50.9 ,	46,
54 ,	47.4 ,	43,
55 ,	48.2 ,	41,
73 ,	60.6 ,	46,
66 ,	59.1 ,	54,
64 ,	59.3 ,	55,
70 ,	60.9 ,	52,
70 ,	60.9 ,	55,
59 ,	50.9 ,	48,
59 ,	52.4 ,	46,
61 ,	53.8 ,	46,
63 ,	58.1 ,	54,
64 ,	58.1 ,	52,
61 ,	56.9 ,	54,
59 ,	55.5 ,	52,
57 ,	53.6 ,	50,
55 ,	52.6 ,	50,
61 ,	56.4 ,	52,
70 ,	62.6 ,	59,
68 ,	62.5 ,	59,
72 ,	64.3 ,	61,
63 ,	61.9 ,	59,
70 ,	62.4 ,	59,
77 ,	65.7 ,	59,
70 ,	65.9 ,	63, 
70 ,	66.2 ,	64,
66 ,	64.5 ,	63,
64 ,	63.6 ,	63,
68 ,	64.6 ,	63,
77 ,	68.9 ,	57,
77 ,	68.5 ,	61,
70 ,	63.4 ,	59,
73 ,	66.8 ,	61,
68 ,	64.6 ,	61,
68 ,	61.3 ,	57,
70 ,	62.7 ,	55,
66 ,	61.2 ,	55,
70 ,	60.9 ,	54,
75 ,	66.8 ,	57,
68 ,	65.1 ,	63,
79 ,	69.0 ,	61,
77 ,	69.5 ,	63,
72 ,	65.0 ,	59,
75 ,	66.7 ,	57,
75 ,	69.2 ,	64,
72 ,	67.0 ,	63,
61 ,	58.4 ,	55,
73 ,	63.6 ,	55,
79 ,	69.6 ,	63,
72 ,	68.3 ,	64,
77 ,	69.0 ,	63,
79 ,	71.5 ,	64,
75 ,	70.4 ,	64,
75 ,	69.5 ,	63,
73 ,	66.9 ,	61,
73 ,	66.7 ,	63]

temp_2013=[63, 	57.2, 	48,
54, 	47.4 ,	43,
46, 	43.4 ,	39,
46, 	43.5 ,	39,
55, 	46.1 ,	37,
57, 	49.1 ,	41,
63, 	56.0 ,	46,
72, 	64.4 ,	55,
70, 	59.4 ,	52,
75, 	57.6 ,	43,
52, 	45.1 ,	39,
59, 	51.0 ,	41,
70, 	62.6 ,	55,
57, 	46.8 ,	43,
55, 	48.0 ,	41,
63, 	53.4 ,	46,
61, 	54.1 ,	46,
68, 	64.2 ,	57,
72, 	63.5 ,	55,
66, 	62.2 ,	57,
61, 	52.5 ,	46,
64, 	55.2 ,	46,
61, 	55.8 ,	52,
54, 	51.2 ,	48,
52, 	49.6 ,	45,
52, 	47.6 ,	43,
50, 	46.9 ,	43,
68, 	55.4 ,	50,
63, 	59.4 ,	55,
54, 	48.5 ,	46,
46, 	45.6 ,	45, 
57, 	50.7 ,	45,
55, 	52.0 ,	48,
61, 	56.1 ,	52,
68, 	60.3 ,	52,
70, 	62.4 ,	55,
68, 	62.3 ,	59,
72, 	64.1 ,	54,
66, 	57.6 ,	50,
70, 	61.6 ,	52,
64, 	56.8 ,	52,
54, 	48.8 ,	45,
59, 	52.2 ,	43,
59, 	52.5 ,	46,
68, 	61.4 ,	54,
66, 	61.9 ,	54,
68, 	59.7 ,	52,
72, 	68.7 ,	66,
72, 	66.8 ,	61,
66, 	58.1 ,	52,
52, 	48.6 ,	45,
50, 	45.3 ,	43,
61, 	51.9 ,	45,
64, 	56.8 ,	48,
64, 	62.2 ,	61,
68 ,	62.2 ,	55,
70, 	62.9 ,	54,
68, 	58.5 ,	50,
66, 	60.7 ,	54,
70, 	62.0 ,	55,
70, 	67.0 ,	63,
61, 	57.0, 	54,
63, 	56.2 ,	52,
64, 	57.7 ,	52,
68, 	59.7, 	54,
70, 	63.0 ,	57,
75, 	68.1 ,	59,
66, 	60.7, 	52,
70, 	60.7 ,	50,
73, 	65.8 ,	55,
79, 	69.5, 	61,
70, 	66.4 ,	64,
77, 	66.0 ,	61,
75, 	65.6, 	59,
81, 	70.3 ,	63,
77, 	67.6, 	63,
75, 	67.8 ,	61,
72, 	62.9 ,	57,
75 ,	67.0, 	57,
73, 	68.0 ,	63,
70, 	65.6 ,	63,
77, 	71.5 ,	68,
79, 	70.3 ,	66,
75 ,	70.3 ,	64,
79, 	71.0 ,	63,
70 ,	63.3 ,	57,
77, 	70.4 ,	64,
77, 	70.8 ,	66,
77 ,	71.2 ,	68,
75, 	71.0 ,	68,
72, 	69.5 ,	68,
75, 	70.3 ,	64]
temp_2014=[52 ,	47.7 ,	45,
45, 	42.8 ,	41,
50, 	43.9 ,	41,
48, 	43.3 ,	37,
50, 	44.5 ,	41,
48, 	42.5 ,	39,
46, 	39.2 ,	36,
48, 	41.9 ,	34,
48, 	42.7 ,	39,
46, 	41.3 ,	36,
50, 	42.7 ,	34,
61, 	51.2 ,	39,
64, 	54.5 ,	46,
55, 	49.9 ,	45,
52, 	45.6 ,	41,
64, 	52.2 ,	39,
61, 	51.9 ,	45,
68, 	59.4 ,	46,
55, 	51.2 ,	50,
50, 	46.3 ,	45,
57, 	50.1 ,	45,
55, 	48.3 ,	41,
61, 	53.2 ,	45,
64, 	55.2 ,	46,
72, 	60.4 ,	52,
66, 	59.9 ,	54,
61, 	54.9 ,	50,
68, 	56.4 ,	46,
70, 	63.5 ,	57,
66, 	61.4 ,	57,
68, 	59.8 ,	55, 
61, 	55.5 ,	52,
64, 	57.3 ,	50,
59, 	56.7 ,	55,
70, 	59.1 ,	54,
57, 	51.7 ,	46,
54, 	47.4 ,	45,
59, 	51.0 ,	41,
70, 	57.5 ,	46,
64, 	58.5 ,	54,
72, 	60.0 ,	54,
63, 	54.3, 	46,
64, 	56.5, 	48,
63, 	56.9, 	52,
61, 	56.5, 	54,
68, 	60.3, 	50,
73, 	63.3, 	55,
68, 	61.5, 	55,
63, 	53.8, 	48,
61, 	52.9, 	48,
55, 	50.8, 	45,
63, 	55.4, 	50,
63, 	57.6, 	55,
66, 	59.8, 	54,
66, 	60.4, 	55,
72, 	62.9, 	55,
72, 	63.7, 	57,
70, 	63.3, 	57,
72, 	64.5, 	59,
68, 	63.8, 	61,
64, 	62.0, 	57,
73, 	67.1, 	64,
75, 	68.0, 	63,
77, 	68.4, 	63,
73, 	63.9, 	59,
68, 	63.3, 	59,
63, 	59.0, 	55,
68, 	60.9, 	54,
70, 	65.1, 	59,
79, 	67.4, 	59,
75, 	67.2, 	55,
75, 	68.3, 	63,
73, 	67.1, 	61,
70, 	65.0, 	63,
79, 	69.2, 	63,
68, 	65.9, 	63,
75, 	69.7, 	64,
77, 	68.9, 	61,
77, 	67.8, 	61,
75, 	68.0, 	63,
77, 	69.0, 	64,
66, 	63.8, 	61,
72, 	64.1, 	59,
70, 	63.3, 	57,
75, 	68.1, 	59,
77, 	71.0, 	68,
75, 	70.1, 	66,
77, 	68.2, 	64,
81, 	70.6, 	66,
79, 	71.6, 	66,
81, 	72.3, 	66,
86, 	74.7, 	66]

temp_2015 =[50 ,	45.8, 	43,
55, 	50.0, 	45,
46, 	43.7, 	41,
61, 	47.6, 	39,
55, 	49.9, 	45,
50, 	45.9, 	43,
46, 	43.9, 	41,
52, 	48.1, 	43,
50, 	49.5, 	48,
55, 	46.4, 	36,
52, 	44.1, 	34,
55, 	47.7, 	41,
57, 	49.4, 	39,
52, 	48.9 ,	46,
54, 	48.5, 	43,
55, 	52.2, 	48,
68, 	57.4, 	50,
66, 	57.6, 	52,
57, 	53.9, 	52,
55, 	52.5 ,	50,
55, 	50.3, 	48,
63, 	55.6, 	46,
57, 	51.3, 	43,
57, 	47.9, 	39,
55, 	47.7, 	41,
55, 	49.4 ,	43,
63, 	53.9, 	43,
66, 	57.0, 	46,
68, 	59.4, 	54,
68, 	59.9, 	52,
70, 	61.8, 	52, 
66, 	57.2 ,	50,
59, 	51.6, 	46,
70, 	62.4, 	54,
70, 	57.1 ,	50,
52, 	50.9, 	50,
70, 	57.0, 	54,
57, 	52.0, 	46,
48, 	41.7, 	37,
52, 	45.8, 	41,
54, 	48.0 ,	45,
59, 	52.4, 	48,
61, 	54.7, 	50,
54, 	50.9, 	48,
57, 	54.1, 	50,
68, 	60.4, 	55,
70, 	60.1 ,	52,
72, 	64.4, 	54,
64, 	58.3, 	50,
63, 	58.2, 	52,
68, 	64.7, 	59,
66, 	63.7, 	59,
66, 	60.0, 	55,
72, 	63.1, 	55,
70, 	64.2, 	59,
68, 	62.9, 	59,
72, 	63.5, 	57,
72, 	66.2, 	57,
75, 	67.8, 	61,
72 ,	66.9, 	61,
72, 	65.8, 	61,
77, 	68.7, 	63,
79, 	68.6, 	61,
79, 	69.1, 	64,
77, 	69.9, 	66,
70, 	65.7, 	63,
73, 	64.8, 	55,
73, 	66.8, 	61,
75, 	68.3, 	63,
70, 	65.8, 	63,
77, 	67.7, 	61,
72, 	63.0 ,	55,
72, 	67.4, 	63,
79, 	72.4, 	66,
79, 	72.6, 	64,
79, 	71.5, 	68,
73, 	69.8, 	68,
81, 	70.9 ,	66,
81, 	70.9, 	64,
77, 	68.2, 	63,
77, 	70.8, 	66,
77, 	66.8, 	59,
73, 	68.8, 	63,
75, 	69.4, 	63,
73, 	70.0, 	66,
73, 	69.9 ,	66,
82, 	71.7, 	64,
81, 	74.2, 	68,
81, 	71.0, 	66,
72, 	69.6, 	66,
81, 	73.4, 	68, 
84, 	76.0   ,72 ]

temp_2016 = [48 ,	42.2 ,	37,
52, 	44.4 ,	37,
57, 	49.3 ,	41,
57, 	50.9 ,	46,
59, 	51.7 ,	48,
63, 	56.2 ,	50,
63, 	58.5 ,	57,
66, 	57.3 ,	55,
59, 	49.3 ,	41,
46, 	43.9 ,	41,
45, 	41.7 ,	39,
45, 	44.7 ,	43,
48, 	45.3 ,	43,
46, 	43.9 ,	41,
55, 	48.1 ,	41,
55, 	48.8 ,	43,
66, 	55.2 ,	48,
66, 	59.5 ,	48,
63, 	59.2 ,	57,
61, 	55.7 ,	52,
52, 	49.5 ,	46,
61, 	52.7 ,	45,
59, 	53.3 ,	46,
48, 	44.3 ,	41,
54, 	48.1 ,	41,
50, 	46.3 ,	43,
57, 	50.6 ,	46,
55, 	52.4 ,	50,
63, 	54.4 ,	46,
68, 	60.0 ,	52,
66, 	59.2 ,	54, 
61, 	55.6 ,	48,
55, 	51.3 ,	48,
63, 	55.4 ,	50,
72, 	61.9, 	52,
54, 	51.8, 	50,
66, 	57.7 ,	48,
70, 	58.4, 	54,
68, 	60.5, 	55,
72, 	63.2, 	57,
70, 	62.8, 	55,
61, 	53.8, 	48,
57, 	50.6, 	45,
66, 	59.1, 	54,
63, 	59.2, 	57,
66, 	61.8, 	55,
66, 	60.3 ,	54,
72, 	66.5, 	63,
75, 	66.3, 	59,
70, 	61.0, 	55,
66, 	58.8, 	52,
70, 	63.6, 	57,
66, 	61.0, 	57,
70, 	63.4, 	57,
66, 	60.8, 	55,
70, 	63.3, 	57,
75, 	66.8, 	61,
68, 	64.0, 	61,
61, 	58.7, 	57,
68, 	61.6, 	55,
68, 	61.4, 	54, 
75, 	63.6, 	55,
68, 	62.4, 	57,
75, 	67.9, 	61,
79, 	70.1, 	66,
79, 	72.5, 	66,
68, 	65.7, 	63,
73, 	68.3, 	63,
79, 	70.0, 	66,
72, 	64.9, 	61,
72, 	65.3, 	63,
79, 	71.8, 	64,
79, 	70.7, 	64,
77, 	69.8, 	63,
72, 	67.9, 	61,
70, 	62.9, 	57,
75, 	66.9, 	59,
70, 	66.6, 	63,
75, 	67.6, 	61,
72, 	67.3, 	63,
66, 	63.0, 	59,
75, 	65.8, 	59,
81, 	71.7, 	64,
84, 	74.2, 	64,
82, 	74.1, 	68,
75, 	71.5, 	68,
77, 	72.5, 	70,
72, 	68.9, 	64,
73, 	67.5, 	63,
81, 	71.0, 	64,
72, 	64.6, 	61,
75, 	68.3, 	63]

temp_2017=[54, 	44.2, 	37,
48, 	45.6 ,	45,
61, 	49.9 ,	41,
55, 	48.9 ,	43,
59, 	51.1 ,	43,
54, 	49.5 ,	48,
50, 	44.9 ,	39,
50, 	43.4 ,	37,
54, 	46.1 ,	36,
55, 	48.0 ,	41,
52, 	46.5 ,	43,
50, 	46.0 ,	41,
52, 	48.1 ,	45,
52, 	47.4 ,	45,
46, 	43.0 ,	41,
57, 	47.4 ,	41,
55, 	49.5 ,	43,
57, 	51.0 ,	41,
63, 	54.6 ,	45,
61, 	54.6 ,	46,
55, 	49.9 ,	46,
57, 	50.3 ,	45,
55, 	50.1 ,	45,
54, 	48.3 ,	43,
52, 	46.3 ,	41,
48, 	44.7 ,	43,
46, 	42.5 ,	37,
54, 	48.4 ,	43,
55, 	48.5 ,	43,
61, 	53.3 ,	45,
57, 	49.8 ,	43,
48, 	44.6 ,	43,
54, 	47.4 ,	43,
59, 	50.5 ,	46,
59, 	51.7 ,	45,
64, 	58.2 ,	48,
68, 	63.1 ,	59,
68, 	63.4 ,	61,
64, 	61.3 ,	55,
64, 	58.3 ,	55,
55, 	52.2 ,	48,
52, 	49.3 ,	46,
70, 	58.3 ,	48,
61, 	54.8 ,	50,
66, 	59.1 ,	48,
72, 	66.9 ,	63,
75, 	65.4 ,	57,
72, 	63.5 ,	59,
72, 	66.3 ,	59,
79, 	66.9 ,	57,
66, 	59.4 ,	52,
64, 	60.1 ,	57,
64, 	59.0 ,	52,
64, 	56.6 ,	52,
63, 	58.1 ,	54,
68, 	61.2 ,	54,
64, 	62.5 ,	59,
64, 	58.5 ,	54,
68, 	60.6 ,	55,
73, 	63.0 ,	54,
73, 	64.7 ,	55, 
72, 	64.7 ,	61,
70, 	62.1 ,	57,
70, 	63.0 ,	57,
72, 	65.1 ,	59,
75, 	66.7 ,	61,
79, 	70.0 ,	63,
72, 	67.8 ,	63,
79, 	68.5 ,	59,
70, 	66.6 ,	64,
64, 	62.2 ,	61,
79, 	69.0 ,	61,
79, 	69.2 ,	64,
68, 	64.9 ,	63,
68, 	64.7 ,	61,
70, 	64.0 ,	61,
72, 	65.6 ,	63,
66, 	63.5 ,	63,
72, 	65.1 ,	63,
77 ,	67.2 ,	61,
81, 	71.0 ,	64,
84, 	73.9 ,	64,
79, 	72.6 ,	68,
77, 	71.6 ,	68,
77, 	71.7 ,	68,
75, 	70.5 ,	68,
70, 	66.5 ,	64,
79, 	69.5 ,	64,
77,	    69.6 ,	64,
77, 	70.6 ,	64,
79, 	72.9 ,	66,
77, 	73.8 ,	72]

temp_2018=[72, 	55.5, 	46,
55, 	50.7, 	45,
57, 	50.2, 	43,
68, 	58.6, 	48,
66, 	61.7, 	50,
50, 	46.9, 	43,
46, 	42.3, 	39,
50, 	44.0, 	41,
64, 	56.7, 	45,
50, 	46.3, 	43,
57, 	49.2, 	43,
57, 	49.7, 	43,
63, 	53.8, 	45,
68, 	58.1, 	46,
70, 	61.5, 	52,
64, 	57.7, 	45,
50, 	45.1, 	39,
63, 	53.1, 	45,
64, 	59.4, 	52,
54, 	48.5, 	46,
45, 	40.6, 	37,
57, 	48.5, 	41,
59, 	53.9, 	50,
57, 	53.0, 	50,
64, 	55.8, 	48,
66, 	58.4, 	50,
68, 	59.9, 	54,
72, 	62.3, 	55,
73 ,	63.5, 	55,
64, 	57.8, 	54,
63, 	56.7, 	48,
68, 	61.6, 	54,
72, 	63.0, 	57,
70, 	63.3, 	59,
77, 	68.7, 	59,
59 ,	56.1, 	54,
72, 	63.3, 	54,
70, 	64.3, 	54,
61, 	55.0, 	50,
68, 	58.4, 	46,
63, 	57.1, 	52,
70, 	64.3, 	57,
77, 	66.4, 	61,
66, 	60.5, 	55,
68, 	60.0, 	55,
75, 	65.0, 	59,
64 ,	59.0, 	55,
61 ,	56.2, 	54,
63 ,	56.3, 	55,
70 ,	61.6, 	54,
75 ,	65.1, 	57,
73 ,	66.9, 	59,
77 ,	68.6, 	61,
66 ,	62.0, 	59,
72 ,	65.4, 	59,
70 ,	66.2, 	64,
73 ,	66.0, 	61,
72 ,	66.0 ,	61,
75 ,	68.3, 	63,
75 ,	68.9, 	61,
75 ,	70.1, 	66,
81 ,	71.2, 	64,
77 ,	68.8, 	64,
79 ,	71.5, 	64,
73 ,	66.8, 	59,
73 ,	65.5, 	57,
77 ,	69.8, 	61,
70 ,	65.0, 	59,
59 ,	57.4, 	55,
59 ,	55.5, 	54,
59 ,	55.5, 	52,
70 ,	62.3, 	52,
73 ,	65.8 ,	57,
72 ,	65.3, 	59,
84 ,	70.0, 	61,
79 ,	73.0, 	64,
81 ,	74.3 ,	68,
79 ,	74.0, 	70,
77 ,	71.7, 	68,
75, 	70.6, 	63,
66 ,	62.1, 	57,
77 ,	68.8, 	61,
79 ,	69.7, 	64,
70 ,	65.8, 	63,
77 ,	69.3, 	63,
79 ,	73.5, 	68,
75 ,	71.3, 	68,
79 ,	69.9, 	64,
77 ,	71.5, 	66,
81 ,	74.2, 	70,
75 ,	70.0,	64,
72 ,	68.4, 	64]

temp_2019=[54,	49.2, 	46,
55, 	50.5 ,	46,
50, 	45.7, 	43,
50, 	47.1 ,	45,
55, 	49.4, 	43,
57, 	52.9, 	50,
52, 	48.6, 	46,
52, 	46.7, 	41,
63, 	52.7, 	41,
59, 	54.6, 	50,
61, 	54.4, 	50,
59, 	54.4, 	50,
64, 	55.5, 	43,
54, 	48.7, 	43,
57, 	50.5, 	45,
54, 	49.4, 	45,
59, 	49.7, 	43,
57, 	50.5, 	45,
64, 	56.7, 	50,
66, 	58.9, 	52,
70, 	63.4, 	55,
70, 	62.0, 	50,
48, 	45.1, 	41,
57, 	48.2, 	37,
59, 	51.5, 	43,
55, 	53.7, 	52,
66, 	59.0, 	48,
61, 	56.8, 	52,
52, 	46.4, 	45,
52, 	48.8, 	46,
57, 	51.6, 	48, 
55, 	48.5, 	43,
54, 	46.5, 	41,
55, 	48.2, 	41,
59, 	52.2, 	43,
72, 	62.5, 	50,
68, 	60.9, 	55,
68, 	59.7, 	54,
59, 	50.6, 	46,
61, 	53.1, 	45,
52, 	45.7, 	41,
61, 	51.5, 	41,
52, 	49.4, 	48,
61, 	53.3, 	46,
64, 	59.2 ,	54,
70, 	60.8, 	55,
66, 	59.4, 	54,
68, 	59.9, 	54,
72, 	62.8, 	57,
72, 	65.1, 	61,
64, 	57.6, 	52,
72, 	63.5, 	57,
73, 	65.3, 	61,
73, 	65.1, 	61,
72, 	67.1, 	64,
73, 	67.7, 	64,
68, 	55.8, 	52,
55, 	51.6, 	48,
63, 	54.4, 	45,
64, 	59.1, 	52,
64, 	59.6, 	57,
70, 	63.3, 	59,
72, 	64.4, 	61,
77, 	65.1, 	57,
72, 	65.8, 	61,
75, 	65.9, 	57,
72, 	66.9, 	63,
68, 	61.0, 	57,
70, 	62.2, 	50,
73, 	67.5, 	64,
77, 	68.5, 	61,
79, 	69.1, 	61,
68, 	63.6, 	59,
72, 	63.0, 	57,
68, 	64.6, 	63,
73, 	66.0, 	61,
75, 	68.0, 	64,
75, 	68.3, 	64,
77, 	68.8, 	64,
75, 	68.4, 	63,
75, 	68.7, 	63,
72, 	69.0, 	66,
73, 	66.8, 	61,
79, 	70.3, 	63,
84, 	73.1, 	66,
88, 	75.3, 	68,
88, 	77.0, 	68,
86, 	76.7, 	68,
73, 	70.0, 	68,
75, 	68.4, 	66,
77, 	72.8, 	66,
75, 	71.4, 	68]

temp_2020=[61, 	53.9, 	48,
54, 	46.9 ,	45,
61, 	53.8 ,	45,
54, 	49.5 ,	48,
55, 	50.0 ,	46,
57, 	49.7 ,	45,
50, 	46.6 ,	45,
50, 	47.0 ,	45,
64, 	55.3 ,	48,
66, 	58.2 ,	55,
66, 	61.3 ,	52,
59, 	52.7 ,	46,
66, 	56.4 ,	48,
57, 	44.1 ,	37,
52, 	46.5 ,	39,
57, 	46.8 ,	39,
54, 	47.4 ,	39,
63, 	54.3 ,	43,
70, 	58.5 ,	48,
64, 	60.3 ,	54,
70, 	58.8 ,	48,
72, 	63.0 ,	55,
57, 	51.6 ,	48,
54, 	48.6 ,	43,
57, 	50.7 ,	43,
66, 	55.7 ,	48,
72, 	62.4 ,	50,
70, 	59.6 ,	48,
46, 	41.7 ,	37,
52, 	47.0 ,	41,
55, 	51.3 ,	48,
57, 	54.0 ,	52,
66, 	58.8 ,	52,
64, 	56.0 ,	48,
70, 	61.0 ,	52,
63, 	53.4 ,	48,
63, 	54.4 ,	46,
61, 	54.7 ,	50,
66, 	58.9 ,	50,
63, 	57.3 ,	48,
61, 	52.8 ,	46,
61, 	53.1 ,	46,
57, 	51.9 ,	48,
52, 	49.6 ,	45,
61, 	53.4 ,	45,
66, 	56.8 ,	46,
59, 	54.4 ,	50,
59, 	53.6 ,	48,
64, 	58.3 ,	55,
70, 	59.5 ,	52,
52, 	50.8 ,	48,
66, 	58.8 ,	52,
63, 	57.2 ,	52,
61, 	54.3 ,	46,
63, 	55.5 ,	48,
63, 	56.0 ,	46,
73, 	66.2 ,	61,
64, 	56.7 ,	52,
66, 	56.6 ,	52,
68, 	60.2 ,	52,
72, 	64.8 ,	57 ,
75, 	68.2 ,	64,
77, 	68.3 ,	61,
77, 	69.7 ,	64,
68, 	65.6 ,	64,
77, 	66.5 ,	64,
64, 	61.9 ,	61,
72, 	63.4 ,	59,
70, 	62.1 ,	55,
70, 	65.9 ,	63,
75, 	69.5 ,	66,
75, 	72.3 ,	68,
77, 	70.0 ,	66,
82, 	73.1 ,	68,
75, 	69.7 ,	64,
77, 	70.5 ,	64,
70, 	65.5 ,	63,
79, 	70.1 ,	63,
70, 	67.2 ,	66,
68, 	66.5 ,	59,
59, 	57.0 ,	54,
59, 	56.3 ,	54,
63, 	60.0 ,	57,
73, 	64.8 ,	61,
79, 	69.2 ,	63,
77, 	70.9 ,	66,
73, 	69.7 ,	68,
75, 	70.8 ,	68,
73, 	69.2 ,	64,
79, 	69.8 ,	61,
79, 	72.2 ,	66,
77, 	71.1 ,	68 ]

temp_2021=[64 ,	53.8, 	46,
68, 	57.5 ,	48,
50, 	45.8 ,	41,
54, 	49.1 ,	43,
59, 	54.2 ,	50,
64, 	55.6 ,	52,
52, 	46.5 ,	45,
46, 	44.9 ,	43,
54, 	49.6 ,	45,
64, 	55.1 ,	48,
59, 	50.5 ,	45,
59, 	54.6 ,	48,
57, 	53.9 ,	50,
64, 	56.5 ,	48,
64, 	56.0 ,	48,
70, 	61.6 ,	48,
64, 	57.6 ,	50,
63, 	53.6 ,	46,
61, 	55.5 ,	52,
63, 	57.9 ,	54,
68, 	64.0 ,	59,
63, 	55.7 ,	48,
59, 	51.7 ,	43,
66, 	57.7 ,	46,
64, 	58.7 ,	55,
66, 	58.0 ,	52,
64, 	57.2 ,	50,
66, 	62.5 ,	59,
72, 	62.0 ,	57,
68, 	63.4 ,	59,
66, 	63.4 ,	59, 
66, 	61.8 ,	57,
66, 	61.3 ,	57,
68, 	62.2 ,	57,
72, 	65.1 ,	63,
66, 	59.6, 	50,
57, 	52.0, 	48,
63, 	56.1, 	50,
64, 	57.4, 	54,
61, 	54.1, 	48,
55, 	50.5, 	46,
61, 	55.0, 	50,
66, 	58.4, 	54,
68, 	60.7, 	55,
70, 	62.1, 	52,
57, 	51.6, 	46,
63, 	57.4, 	52,
66, 	60.4, 	59,
72, 	63.9, 	55,
68, 	60.5, 	54,
72, 	62.9, 	54,
73, 	65.3, 	59,
81, 	66.7 ,	55,
63, 	56.5, 	50,
66, 	59.7, 	52,
72, 	64.0, 	57,
64, 	56.2, 	50,
68, 	58.8, 	52,
73, 	66.0, 	55,
66, 	63.9, 	61,
73, 	66.8, 	59, 
72, 	64.5, 	61,
72, 	64.9, 	57,
70, 	61.8, 	54,
75, 	65.4, 	57,
70, 	67.2, 	64,
70, 	64.9, 	61,
66, 	63.5, 	59,
73, 	67.1, 	57,
82, 	73.4, 	66,
75, 	68.8, 	63,
66, 	64.3, 	63,
68, 	64.2, 	61,
64, 	62.4, 	59,
79, 	68.8, 	63,
79, 	70.0, 	66,
73, 	68.6, 	66,
77, 	72.5, 	68,
75, 	69.7, 	64,
66 ,	63.8, 	61,
70, 	67.4, 	64,
77, 	72.1, 	66,
73, 	69.1, 	66,
75, 	67.3, 	61,
81, 	70.8, 	66,
82, 	71.5, 	64,
73, 	70.0, 	66,
68, 	64.2, 	61,
77, 	68.9, 	63,
79, 	73.4, 	70,
75, 	70.8, 	68,
77, 	69.9, 	66]

def  split_list(l):
    interesting_elements = l[1::3]
    return interesting_elements


    
all_temps=temp_2010 + temp_2011 + temp_2012 + temp_2013 + temp_2014 + temp_2015 + temp_2016 + temp_2017 + temp_2018 + temp_2019 + temp_2020    

all_temp_=split_list(all_temps)

tokyo_sakura = pd.DataFrame(list(zip(dates_, all_temp_)), columns=["date", "temperature"])
tokyo_sakura["flower_status"]=0
tokyo_sakura.head()

start_dates=   ["2010/3/23", "2011/3/30","2012/4/3","2013/3/17","2014/3/26","2015/3/25","2016/3/20","2017/3/23","2018/3/19","2019/3/22","2020/3/14"]

full_dates=    ["2010/4/1", "2011/4/8",  "2012/4/9","2013/3/25","2014/3/31","2015/3/30","2016/4/1","2017/4/6","2018/3/26","2019/4/3","2020/3/24"]

scatter_dates= ["2010/4/8", "2011/4/14", "2012/4/13","2013/3/29","2014/4/4","2015/4/6","2016/4/8","2017/4/12","2018/4/2","2019/4/6","2020/4/2"]

def fill_dates(dates_list, character):
    for date in dates_list:
        tokyo_sakura.loc[tokyo_sakura["date"]==date, "flower_status"] = character
        
list_character={"bloom starts":start_dates, "full":full_dates, "scatter":scatter_dates}

for character in list_character:
    list_=list_character[character]
    fill_dates(list_, character)
    
tokyo_sakura.flower_status.value_counts()

def fahrenheit_to_Celsius(dataframe, column):
    dataframe[column]=dataframe[column].apply(lambda x: (x-32)*5/9)
fahrenheit_to_Celsius(tokyo_sakura, "temperature")
tokyo_sakura["temperature"]=tokyo_sakura["temperature"].round(1)

tokyo_sakura.to_csv("tokyo_sakura.csv", index=False)
        

#%%
# data 2022 march 1. to march 10. for prediction

temp_2022 = [11, 12, 10, 8, 11, 9, 9, 6, 8, 10]

for i in range(82):
    temp_2022.append(0)
    
dates_2022 = make_dates(2022,2022)
temp_tokyo_2022 =  pd.DataFrame(list(zip(dates_2022,temp_2022)), columns=["date", "temperature"])

temp_tokyo_2022.to_csv("tokyo_2022.csv", index=False)
