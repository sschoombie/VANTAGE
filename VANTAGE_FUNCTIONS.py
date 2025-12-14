#TO-DO 2024-2-10
# - Add a function to choose the label name
# - Export all columns if more than one label column is specified - currently only 'current_label_col' will be exported.
#add menu entry to load a folder and automatically load all the data.
#Modify so that only video can be loaded and exported as a df
#2025-10-09 Add a function to load MOV files without transforming
#Add function to load CATS tag data without formatting
#Fix sqaushed image with long videos (> 1 hr)

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

#from scipy.signal import savgol_filter
from scipy import signal

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

import cv2
from PIL import Image
from PIL import ImageTk

from scipy.io.wavfile import read
# from tensorflow.keras.models import  model_from_json
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
#from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg#,
    #NavigationToolbar2Tk
)
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider#, Button

import pandas as pd
from datetime import datetime,timedelta
import time

import sys
import numpy as np
import os
import glob
import math

# import keyboard
from pytz import utc

import subprocess
import json

from Data_Frame import Data_Frame

#This is only for YOLO models
try:
    from ultralytics import YOLO
except:
    print("ultralytics lib not found - YOLO models won't be available")


class Menu_functions_FILE(Data_Frame):

    def load_csv(self,dat):
############
# load_csv #
############
# This function reads a selected csv file into low_memory
# The file is the raw data - this file will not be modified, rather additonal files are created to store analysed data.
        #Check to see if view_only mode is selected
        if dat.view_only:
            filename = dat.filename
        else:
        #otherwise open a dialog box to select a file from
            file_types = (('csv files', '*.csv'),('All files', '*.*'))
            filename = fd.askopenfilename(
                        title='Open a file',
                        # initialdir=dat.wd,
                        filetypes=file_types)

        #If the selected file is in a data file
        if (len(filename) > 0):
            print("LOADING.... " + filename)
            # print("Standard file")
            dat.filename = filename

            #Create a new window to show a prompt to wait
            #New temp window
            wtemp = tk.Toplevel(self)
            wtemp.title("Loading data")
            wtemp.geometry("400x200")
            wtemp.attributes('-topmost',True)
            #Prompt text
            label = tk.Label(wtemp, text="Please wait - this may take a while")  # create the label
            label.pack()  # add the label to the window
            self.update()

            #Read the data
            dat.df = pd.read_csv(filename,low_memory=False)

            #Check to see if we loaded any data
            if(len(dat.df.columns) == 1):
                #Otherwise we try tab delimiter
                print("Tab delimited")
                dat.df = pd.read_csv(filename,sep='\t',low_memory=False)

            dat.nrow = np.array(dat.df).shape[0]

            if(dat.current_label_col in dat.df):
                pass
            else:
                dat.df[dat.current_label_col] = 0

            wtemp.destroy() #Remove the prompt window

            #If there is no "Timestamp" column, choose one
            print(dat.time_col_string)
            try:
                dat.time_col = dat.df.columns.get_loc(dat.time_col_string)
            except:


                #Create a new promt window to select the data column from
                wtemp = tk.Toplevel(self)
                wtemp.title("Please select the DateTime column")
                wtemp.geometry("600x400")
                wtemp.attributes('-topmost',True)

                #DateTime column selection
                dat.time_col = tk.StringVar()
                dat.time_col.set('Select DateTime column')
                print((dat.time_col.get()))
                # Column name options
                column_names = dat.df.columns.tolist()
                column_menu = tk.OptionMenu(wtemp, dat.time_col, *column_names)
                tk.Label(wtemp, text="Choose a column").grid(row=0, column=0)
                column_menu.grid(row=1, column =0)

                #Function when the "Apply button is pressed"
                #Once the column name and format has been chosen we convert the column to POSIX
                def tapply():
                    print(dat.time_col.get())
                    s_col = dat.time_col.get()
                    #Save the column string
                    dat.time_col_string = s_col
                    # #Save the column number of Timestamp
                    dat.time_col = dat.df.columns.get_loc(s_col)

                    ##        dat.df = np.array(dat.df)
                    dat.temp_done = True
                    wtemp.destroy()

                #Create the button and add the
                btnApply = tk.Button(wtemp,text = "Apply",command = tapply)
                btnApply.grid(row = 2, column = 0)

                #wait for a column to be chosen
                dat.temp_done = False
                while dat.temp_done == False:
                    self.update()

            #We try a range of DateTime formats as this may vary between files
            s_format =  "%Y/%m/%d %H:%M:%S.%f"
            try:
                dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = s_format,utc =utc)
            except:
                try:
                    s_format =  "%Y-%m-%d %H:%M:%S.%f"
                    dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = s_format,utc =utc)
                except:
                    try:
                        s_format =  "%d/%m/%Y %H:%M:%S.%f"
                        dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = s_format,utc =utc)
                    except:
                        try:
                            s_format =  "%Y-%m-%d %H:%M:%S.%f"
                            dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = s_format,utc =utc)
                        except:
                            try:
                                s_format =  "%d.%m.%Y %H:%M:%S.%f"
                                dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = s_format,utc =utc)
                            except:
                                dat.df[dat.time_col_string] = pd.to_datetime(dat.df[dat.time_col_string],format = "mixed",utc =utc)
            print(dat.df.loc[0,dat.time_col_string])



            #Get the sampling frequency of the data
            date_diffs = dat.df[dat.time_col_string].diff()
            dat.frequency = int(round(1/(date_diffs).mean().total_seconds(),0))
            print("Sampling rate is: " + str(dat.frequency))

            #############################
            #Load data already processed
            #############################
            ###############
            # 1- OUT file #
            ###############
            #FInd the file name
            pre,ext = os.path.splitext(dat.filename)
            out_file = pre + "_OUT.csv"

            #Create the OUT file headings
            dat.df["vid"] = "NA"
            dat.df["frame"] = "NA"
            dat.df["vid_time"] = "NA"
            dat.df["cfilter"] = 0
            if('BEHAV' in dat.df):
                pass
            else:
                dat.df['BEHAV'] = 0

            #Check if the file exists
            if os.path.exists(out_file):
                temp_df = pd.read_csv(out_file)
                # dat.df = dat.df.join(temp_df)
                dat.df.iloc[:,dat.df.columns.get_loc(dat.current_label_col)] = temp_df.iloc[:,temp_df.columns.get_loc(dat.current_label_col)]

                dat.df.iloc[:,dat.df.columns.get_loc("vid")] = temp_df.iloc[:,temp_df.columns.get_loc("vid")]

                dat.df.iloc[:,dat.df.columns.get_loc("frame")] = temp_df.iloc[:,temp_df.columns.get_loc("frame")]

                dat.df.iloc[:,dat.df.columns.get_loc("vid_time")] = temp_df.iloc[:,temp_df.columns.get_loc("vid_time")]

                if('BEHAV' in temp_df):
                    dat.df.iloc[:,dat.df.columns.get_loc("BEHAV")] = temp_df.iloc[:,temp_df.columns.get_loc("BEHAV")]
                dat.out_file_loaded = True
                try:
                    dat.btn_2['state'] = 'disabled' #Cheatsheet button disable
                except:
                    pass
                print("OUT file loaded")

            ###############
            # 2- DIVES file #
            ###############
            #FInd the file name
            dives_file = pre + "_DIVES.csv"
            #Check if the file exists
            if os.path.exists(dives_file):
                temp_df = pd.read_csv(dives_file)
                dat.df = dat.df.join(temp_df)
                print("DIVES file loaded")

            ###############
            # 3- YOLO file #
            ###############
            #FInd the file name
            yolo_file = pre + "_YOLO.csv"
            #Check if the file exists
            if os.path.exists(yolo_file):
                temp_df = pd.read_csv(yolo_file)
                dat.df = dat.df.join(temp_df)
                print("YOLO file loaded")

            #Create a copy of the data for smoothing and normalizing functions
            ##!! REMOVING THIS FOR THE TIME BEING AS IT TAKES UP TOO MUCH MEMORY!!
            # dat.df_hold = copy.deepcopy(dat.df)

            dat.col_names = dat.df.columns
            c = 0
            for i in dat.col_names:
                exec("dat.c" + str(c) +" = tk.IntVar()")
                exec("dat.cf" + str(c) +" = tk.IntVar()")
                c = c +1

            #Set data flag
            dat.df_loaded = True
            # wtemp.destroy()
            try:
                dat.btn_1['state'] = 'disabled' #Cheatsheet button disable
            except:
                pass

            #Add the first data entry as IMU_date
            array = dat.df.iloc[:,dat.time_col]                  #Define the numeric date column as an array
            dat.IMU_date = array[0]
            print(dat.IMU_date)
            #Plot the data
            Menu_functions_PLOT.plot_dat(self, dat)

    def load_gps(self):
        pass

#     def import_config(self):
#         file_types = (('pkl files', '*.pkl'),('All files', '*.*'))
#         filename = fd.askopenfilename(
#                     title='Open a file',
#                     initialdir=os.getcwd(),
#                     filetypes=file_types)
#         with open(filename, 'rb') as inp:
#             cfg = pickle.load(inp)

#         self.video_offset = cfg.video_offset
#         self.wd = cfg.wd
#         self.v_num = cfg.v_num
#         self.filename = cfg.filename
#         self.dive_num = cfg.dive_num
#         # self.col_names = cfg.col_names
#         self.vid1 = cfg.vid1
#         self.vid_last_date = cfg.vid_last_date
#         self.frame = cfg.frame
#         self.frame_count = cfg.frame_count
#         self.fps = cfg.fps
#         self.vid_start_date = cfg.vid_start_date
#         self.frame_date = cfg.frame_date
#         self.date_set = cfg.date_set
#         self.video_loaded = cfg.video_loaded
#         self.vt_present = cfg.vt_present
#         self.vt = cfg.vt


#         pre,ext = os.path.splitext(self.filename)
#         # in_file = pre + "_OUT.csv"
#         # try:
#         #     self.df = pd.read_csv(in_file)
#         #     self.df.iloc[:,dat.time_col] = pd.to_datetime(self.df.iloc[:,dat.time_col],format = "%Y-%m-%d %H:%M:%S.%f",utc = utc)
#         #     self.df_hold = copy.deepcopy(self.df)
#         # except:
#         #     pass

#         self.vid = cv2.VideoCapture(self.vid1)
#         self.vid.set(1,self.frame)
#         s, self.image = self.vid.read()
#         self.video_loaded = True
#         self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
#         self.image = cv2.resize(self.image,(self.video_width,int(self.video_width/1.5)))
#         self.image = Image.fromarray(self.image)
#         self.image = ImageTk.PhotoImage(self.image)
#         self.panel = tk.Label(image=self.image)
#         self.panel.image = self.image
#         self.panel.pack()
# ##        self.panel.place(x = 700,y = 90)
#         self.panel.place(relx = 0.5,rely = 0.18)
# ##        self.panel.bind("<KeyPress>", lambda x: print(x.char))
#         if(self.date_set == 1):
#             self.ax_main.cla()
#             self.vid_idx_start = (np.abs(self.df.iloc[:,dat.time_col] - self.vid_start_date)).argmin()
#             self.vid_idx_end = self.vid_idx_start + int((self.frame_count/self.fps)*dat.frequency)
#             self.ax_main.plot(self.df.iloc[np.arange(self.vid_idx_start,(self.vid_idx_end),1000),dat.time_col],self.df.iloc[np.arange(self.vid_idx_start,(self.vid_idx_end),1000),self.p_cols])
#             self.temp_vline1 =  self.ax_main.axvline(x = self.df.iloc[self.vid_idx_start,dat.time_col],color = 'blue')
#             self.temp_vline2 =  self.ax_main.axvline(x = self.df.iloc[self.vid_idx_end,dat.time_col],color = 'blue')
#             self.figure_main.canvas.draw()

    def view_data(self,dat):
        if dat.video_loaded:
            wview = tk.Toplevel(self)
            # wview.title("")
            wview.geometry("1000x1000")

            # Create a Treeview widget
            tree = ttk.Treeview(wview, columns=list(dat.df.columns), show="headings")
            # Add columns to Treeview
            for col in dat.df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)

            tree.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            # Create a vertical scrollbar
            vertical_scrollbar = ttk.Scrollbar(wview, orient='vertical', command=tree.yview)
            vertical_scrollbar.grid(row=0, column=1, sticky='ns')
            tree.configure(yscrollcommand=vertical_scrollbar.set)

            # Create a horizontal scrollbar
            horizontal_scrollbar = ttk.Scrollbar(wview, orient='horizontal', command=tree.xview)
            horizontal_scrollbar.grid(row=1, column=0, sticky='ew')
            tree.configure(xscrollcommand=horizontal_scrollbar.set)

            # Configure grid weights for resizing
            wview.grid_rowconfigure(0, weight=1)
            wview.grid_columnconfigure(0, weight=1)

            # for index, row in dat.df[dat.vid_idx_start:dat.vid_idx_end].iterrows():
            #     tree.insert("", "end", values=row)

            data_to_insert = [tuple(row) for row in dat.df[dat.vid_idx_start:dat.vid_idx_end].values.tolist()]
            for item in data_to_insert:
                tree.insert("", "end", values=item)
        else:
            print('load video first')

    def f_quit(self,dat):
        if dat.bexported == False:
            if dat.view_only == False:
                res = tk.messagebox.askyesno("Close without saving","New ANNOTATIONS aren't saved. Save them now?")
                if res:
                    Menu_functions_EXPORT.export_events(self,dat)
                else:
                    res = tk.messagebox.askyesno("Exit","Are you sure?")
                    if res:
                        self.destroy()
                        sys.exit()
                    else:
                        pass
            res = tk.messagebox.askyesno("Exit","Are you sure?")
            if res:
                self.destroy()
                sys.exit()
            else:
                pass
        # elif dat.bconfig == False:
        #     res = tk.messagebox.askyesno("Close without saving","CONFIG not saved, save now?")
        #     if res:
        #         Menu_functions_EXPORT.export_events(dat)
        #     else:
        #         res = tk.messagebox.askyesno("Exit","Are you sure?")
        #         if res:
        #             self.destroy()
        #             sys.exit()
        #         else:
        #             pass
        else:
            res = tk.messagebox.askyesno("Exit","Are you sure?")
            if res:
                self.destroy()
                sys.exit()
            else:
                pass

    def add_synch_button(self,dat):
        dat.btn_SYNC = tk.Button(dat.video_frame,text = "SYNC", command=lambda:Button_functions.set_time(dat))
        # dat.btn_SYNC.place(in_=dat.panel ,relx = 0,rely=  0 )
        dat.btn_SYNC.place(relx = 0.01,rely= 0.95)
        if dat.vid_date_set:
            dat.btn_SYNC.configure(bg = 'green')
        else:
            dat.btn_SYNC.configure(bg = 'red')

        # dat.btn_annotate1 = tk.Button(self,text = "1 - PCE", command=lambda:Button_functions.ann1(self,dat))
        # dat.btn_annotate1.place(in_=dat.panel ,relx = 0.1,rely=  0 )
        # dat.btn_annotate2 = tk.Button(self,text = "2 - PCE?", command=lambda:Button_functions.ann2(self,dat))
        # dat.btn_annotate2.place(in_=dat.panel ,relx = 0.2,rely=  0 )
        # dat.btn_annotate5 = tk.Button(self,text = "5 - Peng", command=lambda:Button_functions.ann5(self,dat))
        # dat.btn_annotate5.place(in_=dat.panel ,relx = 0.3,rely=  0 )
        # dat.btn_annotate6 = tk.Button(self,text = "6 - Swarm", command=lambda:Button_functions.ann6(self,dat))
        # dat.btn_annotate6.place(in_=dat.panel ,relx = 0.4,rely=  0 )
        # dat.btn_annotate7 = tk.Button(self,text = "7 - Ceta", command=lambda:Button_functions.ann7(self,dat))
        # dat.btn_annotate7.place(in_=dat.panel ,relx = 0.5,rely=  0 )
        # dat.btn_annotate9 = tk.Button(self,text = "9 - DARK", command=lambda:Button_functions.ann9(self,dat))
        # dat.btn_annotate9.place(in_=dat.panel ,relx = 0.6,rely=  0 )
        # dat.btn_annotate0 = tk.Button(self,text = "0 - clear", command=lambda:Button_functions.ann0(self,dat))
        # dat.btn_annotate0.place(in_=dat.panel ,relx = 0.7,rely=  0 )
        # dat.btn_annotateGroup = tk.Button(self,text = "Annotate selection", command=lambda:Button_functions.annGroup(self,dat))
        # dat.btn_annotateGroup.place(in_=dat.panel ,relx = 0.8,rely=  0 )

class Menu_functions_VIDEO(Data_Frame):
    def load_avi(self,dat,filename = ""):
        ############
        # load_avi #
        ############
        #Only proceed if the working directory has been set
        if dat.wd is None:
            showinfo(title='No video directory',message="Set the video directory first")
        else:

            #Check to see if view_only mode is selected
            if (dat.view_only):
                filename = dat.vid1
            elif filename != "":
                pass
            else:
                file_types = (('mp4 files', '*.mp4'),('All files', '*.*'))
                filename = fd.askopenfilename(
                            title='Open a file',
                            initialdir=dat.wd,
                            filetypes=file_types)
            if (len(filename) > 0):
                dat.vid1 = filename
                # showinfo(title='Selected File',message=filename)
                dat.vid = cv2.VideoCapture(dat.vid1)
                dat.fps = int(round(dat.vid.get(cv2.CAP_PROP_FPS)))
                dat.frame_count = int(dat.vid.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Frame count: {dat.frame_count}")
                print(f"FPS: {dat.fps}")

                #Get the timestamp time from the video metadata
                #First - Run ffprobe command to get video metadata
                ffprobe_command = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    dat.vid1
                ]

                try:
                    ffprobe_output = subprocess.check_output(ffprobe_command, stderr=subprocess.STDOUT)
                    metadata = json.loads(ffprobe_output.decode('utf-8'))
                    if 'creation_time' in metadata['format']['tags']:
                        creation_time = metadata['format']['tags']['creation_time']
                        dat.vid_last_date = pd.to_datetime(creation_time,utc = utc).timestamp()# - timedelta(seconds = dat.frame_count/dat.fps)
                        print("last date (i.e. creation time): " + str(pd.to_datetime(creation_time,utc = utc)))
                    else:
                        print('no creation time')
                        #Otherwise check the creation date using os.path
                        dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
                except subprocess.CalledProcessError as e:
                    print("Error:", e)
                    dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX


                files = sorted(os.listdir(dat.wd))
                vn = dat.vid1.split("/")[-1]
                dat.v_num = files.index(vn)
                pre,ext = os.path.splitext(filename)
                dat.frame = 0
                video_name = files[dat.v_num]
                vn =  os.path.splitext(video_name)[0]
                # dat.v_num = dat.v_num+1
                # while(((not files[dat.v_num].lower().endswith(".avi")) and (not files[dat.v_num].lower().endswith(".mov")))  and dat.v_num < len(files)):
                #     dat.v_num = dat.v_num + 1
                #
                # print("##############DEBUG##################")
                # print(int(dat.vt["set"][vid_match]))
                print(vn)
                # print(dat.vt['vid'])

                vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)
                if(int(dat.vt["set"][vid_match]) == 1): #if(int(dat.vt["set"][vid_match]) == 1):

                    dat.vid_date_set = 1
                    dat.vid_start_date = dat.vt["vid_start_date"][vid_match]#.values
                    # print(dat.vt["vid_start_date"][vid_match])
                    print(f"Video start date: {dat.vid_start_date}")
                    dat.video_offset = dat.vid_start_date - dat.vt["Timestamp"][vid_match]#.values #timedelta(seconds = int(dat.vt["video_offset"][vid_match]))
                    print("offset: " + str(dat.video_offset))
                    print("File offset: " + str(dat.vt["video_offset"][vid_match]))
                    # print(dat.video_offset)
                else:
                    dat.vid_date_set = 0
                    # dat.vid_start_date = pd.to_datetime(dat.vid_last_date -(dat.frame_count/dat.fps),unit = 's',utc = utc)
                    dat.vid_start_date = pd.to_datetime(dat.vid_last_date ,unit = 's',utc = utc)
                    print("original date: " + str(dat.vid_start_date))
                    vid_start_date = dat.vid_start_date + dat.video_offset
                    print("offset: " + str(dat.video_offset))

                    dat.vid_start_date = vid_start_date
                    print("adjusted date: " + str(dat.vid_start_date))

                dat.frame_date=dat.vid_start_date

                dat.frame = 1
                dat.vid.set(1,dat.frame)
                s, dat.image = dat.vid.read()
                dat.video_loaded = True
                dat.image = cv2.cvtColor(dat.image, cv2.COLOR_BGR2RGB)
                # dat.image = cv2.resize(dat.image,(dat.video_width,int(dat.video_width/1.5)))
                dat.image = cv2.resize(dat.image,(dat.video_frame.winfo_width(),int(dat.video_frame.winfo_width()/1.5)))

                dat.image = Image.fromarray(dat.image)
                dat.image = ImageTk.PhotoImage(dat.image)
                dat.panel = tk.Label(dat.video_frame,image=dat.image)
                dat.panel.image = dat.image
                dat.panel.grid(column = 0, row = 0,sticky = 'nesw')
        #         dat.panel.pack()
        # ##        dat.panel.place(x = 700,y = 90)
        #         dat.panel.place(relx = 0.5,rely = 0.1)

                #Disable the cheatsheet button if it exists
                try:
                    dat.btn_5['state'] = 'disabled' #Cheatsheet button disable
                except:
                    pass

                ########################################
                #Add navigation bar and squashed image
                ########################################

                #Nav Bar
                # define the callback function for the trackbar
                def on_trackbar(ival):
                    # print(ival)
                    dat.frame = int(ival)
                    dat.vid.set(1,int(ival))
                    Keyboard_functions.plot_frame(self,dat)#Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()

                # dat.tbar = tk.Scale(from_=0, to=dat.vid.get(cv2.CAP_PROP_FRAME_COUNT)-1, orient=tk.HORIZONTAL, command=on_trackbar, showvalue= 0,length = dat.video_width)
                dat.tbar = tk.Scale(from_=0, to=dat.vid.get(cv2.CAP_PROP_FRAME_COUNT)-1, orient=tk.HORIZONTAL, command=on_trackbar, showvalue= 1,length = dat.video_frame.winfo_width())
                dat.tbar.place(in_=dat.panel ,relx = 0,rely = 1  )

                #Squashed Img
                try:
                    dat.sq_panel.destroy()
                except:
                    pass
                squash_file = pre + "_squash.jpg"
                if os.path.exists(squash_file):
                    dat.sq_img = cv2.imread(squash_file)
                    dat.sq_img = cv2.cvtColor(dat.sq_img, cv2.COLOR_BGR2RGB)
                    #Make sure the squashed image is rotated correctly
                    if dat.sq_img.shape[0] > dat.sq_img.shape[1]: #height > width
                        dat.sq_img =cv2.rotate(dat.sq_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    print(f"Squashed image width: {dat.sq_img.shape[1]}")

                    # dat.sq_img_re = cv2.resize(dat.sq_img,(dat.video_width,50))
                    dat.sq_img_re = cv2.resize(dat.sq_img,(dat.video_frame.winfo_width(),50))
                    print(f"Squashed image resized width: {dat.sq_img.shape[1]}")
                    dat.sq_img_re = Image.fromarray(dat.sq_img_re)
                    dat.sq_img_re = ImageTk.PhotoImage(dat.sq_img_re)
                    # dat.sq_panel = tk.Button(image=dat.sq_img,command = dat.frame_from_squash(event,self))#,width = dat.video_width,height = 10)
                    dat.sq_panel = tk.Label(image=dat.sq_img_re)
                    dat.sq_panel.image = dat.sq_img_re
                    dat.sq_panel.place(in_ = dat.tbar, relx = 0, rely = 1)
                    #Add a function to the image
                    dat.sq_panel.bind("<Button-1>",lambda event:Button_functions.frame_from_squash(event,dat))
                    dat.squash_loaded = True

                if(dat.date_set == 1 and dat.df_loaded == True):
                    dat.ax_main.cla()

                    dat.vid_idx_start = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin()
                    dat.sub_min = dat.vid_idx_start
                    dat.vid_idx_end = dat.vid_idx_start + int((dat.frame_count/dat.fps)*dat.frequency)

                    #DEBUG trying something with the squash file
                    #Split the colour image into channels (blue, green, red)
                    try:
                        r, g, b = cv2.split(dat.sq_img)
                        #Weonly use the red channel and blur it to reduce noise
                        blur = cv2.GaussianBlur(r,(11,11),0)  #blur image to reduce noise
                        #Save the frame rate
                        fps = dat.fps
                        #Choose a window to smooth the data over
                        window = 10
                        #Specificy the minimum dive duration (i.e. minimum time between dive starts)
                        dive_duration = 20

                        #Get a rollin mean of the red channel values over the chosen window
                        for row in range(blur.shape[0]):
                            image_mean = pd.DataFrame(blur[row,:]).rolling(fps*window).mean()

                        #Create a dataframe of the values and assign frame numbers
                        ts_diff = pd.DataFrame(image_mean.values,columns = ["val"]) #Smoothed values of the red channel
                        ts_diff['raw'] = blur[0,:]
                        ts_diff['frame'] = range(len(ts_diff))
                        ts_diff['dive'] = 0
                        ts_diff['sq_seconds'] = pd.to_timedelta(ts_diff.frame/fps, unit = 's')
                        ts_diff[dat.time_col_string] = dat.vid_start_date + ts_diff['sq_seconds']

                        ts_diff =  ts_diff[[dat.time_col_string,'val']]

                        #Normalize the val column
                        ts_diff['val'] = (ts_diff['val'] - np.min(ts_diff['val'])) / (np.max(ts_diff['val']) - np.min(ts_diff['val']))

                        # print(ts_diff)
                        # print(dat.df.loc[dat.vid_idx_start:dat.vid_idx_start+10,"Timestamp"])
                        # print(f"ts_df length: {len(ts_diff)}")
                        if "val" not in dat.df.columns:
                            dat.df = dat.df.assign(val = 0)

                        merged_df = pd.merge_asof(dat.df, ts_diff, on = dat.time_col_string, direction = 'nearest')

                        dat.df['val'] = merged_df['val_y'].fillna(merged_df['val_x'])

                        # dat.df = pd.merge(dat.df, ts_diff[[dat.time_col_string, 'val']], on=dat.time_col_string, how='left', direction = 'nearest')

                        # print(dat.df.loc[dat.vid_idx_start:dat.vid_idx_start+10,"val"])
                    except:
                        pass

                    #DEBUG end



                    dat.sub_max = dat.sub_min + 1000
                    i_int = 1

                    if (dat.sub_max - dat.sub_min) > 2000:
                        i_int = round((dat.sub_max - dat.sub_min)/2000)
                    # if((dat.vid_idx_end - dat.vid_idx_start) < 1000):
                    #     i_int = 10

                    #Specificy the rows to plot
                    row_to_use = np.arange(dat.vid_idx_start, dat.vid_idx_end, i_int)
                    # Plot each column individually
                    for col in dat.p_cols:
                        # Select the data for the current column
                        x_data = dat.df.iloc[row_to_use, dat.time_col]
                        y_data = dat.df.iloc[row_to_use, col]

                        # Exclude NA values
                        non_na_mask = ~y_data.isna()
                        x_data = x_data[non_na_mask]
                        y_data = y_data[non_na_mask]

                        # Plot the current column
                        dat.ax_main.plot(x_data, y_data, label=dat.df.iloc[:,col].name)

                    # dat.ax_main.plot(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.time_col],dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.p_cols])
                    dat.temp_vline1 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_start,dat.time_col],color = 'blue')
                    dat.temp_vline2 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_end,dat.time_col],color = 'blue')
                    dat.figure_main.canvas.draw()

                    dat.ax_zoom.cla() #Clear axes

                    # Extracting the data for zoom plot
                    #Specificy the rows to plot
                    row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
                    # Plot each column individually
                    for col in dat.p_cols:
                        # Select the data for the current column
                        x_data = dat.df.iloc[row_to_use, dat.time_col]
                        y_data = dat.df.iloc[row_to_use, col]

                        # Exclude NA values
                        non_na_mask = ~y_data.isna()
                        x_data = x_data[non_na_mask]
                        y_data = y_data[non_na_mask]

                        # Plot the current column
                        dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
                    # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
                    dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
                    # dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
                    dat.figure_zoom.canvas.draw() #Redraw the figure
                    self.update()
                #Add button to synchronize time
                Menu_functions_FILE.add_synch_button(self,dat)

                ########################################
                #Add sound if available
                ########################################
                # try:
                #     if os.path.exists(pre+".wav"):
                #         dat.audio_present = True
                #         input_data = read(pre+".wav")
                #         dat.audio = pd.DataFrame()
                #         dat.audio['aud'] = input_data[1]
                #
                #         start_time = dat.vid_start_date
                #
                #         total_seconds = dat.frame_count/dat.fps
                #         time_interval = timedelta(seconds=(total_seconds / (len(dat.audio) - 1)))
                #
                #         # Generate time axis
                #         time_axis = [start_time + i * time_interval for i in range(len(dat.audio))]
                #         dat.audio[dat.time_col_string] = time_axis
                #
                #         print("Audio loaded")
                #     else:
                #         dat.audio_present = False
                # except:
                #     print("no audio found")

    def load_audio(self,dat):
        file_types = (('wav files', '*.wav'),('All files', '*.*'))
        filename = fd.askopenfilename(
                    title='Open a file',
                    filetypes=file_types)

        #Read the wav file and get the sampling frequency
        dat.frequency, input_data = read(filename)
        dat.df = pd.DataFrame()
        dat.df['aud'] = input_data
        #Get the duration in seconds
        aud_dur = dat.df.shape[0] / dat.frequency
        #Create a column for the audio time in seconds
        dat.df = dat.df.assign(seconds = np.linspace(0., aud_dur, dat.df.shape[0]))

        start_time = os.path.getctime(filename) #Time the wav file was created - if the file was extracted from video it will most likely not preserve the time
        # Convert to timestamp
        start_time = datetime.fromtimestamp(start_time)

        #Assign the Timestamp column to the data
        dat.df = dat.df.assign(Timestamp = "NA")
        dat.df.Timestamp = [start_time + timedelta(seconds=sec) for sec in dat.df.seconds]
        s_format =  "%Y-%m-%d %H:%M:%S.%f"
        dat.df.Timestamp = pd.to_datetime(dat.df.Timestamp,format = s_format,utc =utc)

        #Save the column number associated with the Timestamp
        dat.time_col_string = "Timestamp"
        dat.time_col = dat.df.columns.get_loc("Timestamp")

        #Get the number of rows
        dat.nrow = np.array(dat.df).shape[0]
        # print("audio rows:")
        # print(dat.nrow)

        if(dat.current_label_col in dat.df):
            pass
        else:
            dat.df[dat.current_label_col] = 0

        #Video headings
        dat.df["vid"] = "NA"
        dat.df["frame"] = "NA"
        dat.df["vid_time"] = "NA"
        dat.df["cfilter"] = 0

        if('BEHAV' in dat.df):
            pass
        else:
            dat.df['BEHAV'] = 0

        #Create a copy of the data for smoothing and normalizing functions
        ##!! REMOVING THIS FOR THE TIME BEING AS IT TAKES UP TOO MUCH MEMORY!!
        # dat.df_hold = copy.deepcopy(dat.df)

        dat.col_names = dat.df.columns
        c = 0
        for i in dat.col_names:
            exec("dat.c" + str(c) +" = tk.IntVar()")
            exec("dat.cf" + str(c) +" = tk.IntVar()")
            c = c +1

        #Set data flag
        dat.df_loaded = True
        # wtemp.destroy()
        try:
            dat.btn_1['state'] = 'disabled' #Cheatsheet button disable
        except:
            pass

        print("Audio loaded")
    def save_wd(self):
        #Check to see if view_only mode is selected
        if self.view_only:
            foldername = self.wd
        else:
            foldername = fd.askdirectory()
        self.wd = foldername
        try:
            self.btn_4['state'] = 'disabled' #Cheatsheet button disable
        except:
            pass
        self.bwd_set = True
        print("Directory changed: " + self.wd)

        files = sorted(os.listdir(self.wd))
        files_avi = list()

        for f in files:
            if (f.endswith(".mp4") or f.endswith(".mov")):
                files_avi.append(f)

        for f in files:
            if f == "vt.csv":
                print("vt file already present")
                self.vt_present = True
                print('reading: ' + foldername + "/" + f)
                self.vt = pd.read_csv(foldername + "/" + f)
                # print(self.vt.set)
                s_format =  "%Y-%m-%d %H:%M:%S.%f"
                self.vt.iloc[:,1] = pd.to_datetime(self.vt.iloc[:,1],format = "mixed",utc =utc)
                self.vt.iloc[:,2] = pd.to_datetime(self.vt.iloc[:,2],format = "mixed",utc =utc)
                # if(self.vt["set"][1] == 1):
                # if 1 in self.vt.set:
                if (self.vt.set == 1).any():
                    print("date already set")
                    self.date_set = 1

                #Update the vt file with any new video files
                new_rows = []
                for nf in files_avi:
                    if nf not in self.vt['vid'].values:
                        # Prepare a new row as a dictionary
                        new_rows.append({'vid': nf, 'set': 0})

                # Create a DataFrame from the new rows and concatenate it with the existing DataFrame
                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    #Add NA values
                    for col in self.vt:
                        if col not in new_df.columns:
                            new_df[col] = np.nan
                    self.vt = pd.concat([self.vt, new_df], ignore_index=True)
                    print(f"Added {len(new_rows)} new file(s)")
                    self.vt = self.vt.sort_values(by='vid').reset_index(drop=True)
                    self.vt.to_csv(foldername + "/vt.csv",index = False)
                    print("vt file UPDATED")

                # find rows in vt that are not present in files_avi anymore
                removed_files = self.vt[~self.vt['vid'].isin(files_avi)]

                if not removed_files.empty:
                    self.vt = self.vt[self.vt['vid'].isin(files_avi)].reset_index(drop=True)
                    print(f"Removed {len(removed_files)} missing file(s):")
                    print(removed_files['vid'].tolist())
                    self.vt = self.vt.sort_values(by='vid').reset_index(drop=True)
                    self.vt.to_csv(foldername + "/vt.csv",index = False)
                    print("vt file UPDATED")


        if self.vt_present ==False:
            vt = pd.DataFrame(columns = ("vid","vid_start_date","Timestamp","video_offset","set"))
            vt["vid"] = files_avi
            vt["vid_start_date"] = "NA"
            vt["Timestamp"] = "NA"
            vt["video_offset"] = "NA"
            vt["set"] = 0
            vt.to_csv(foldername + "/vt.csv",index = False)
            print("vt file created")
            self.vt = vt
            self.vt_present = True

    #2025-10-10 - Added this for videos from CATS tags with broken indexing in the MOV file
    def fix_video_index(self,dat):
        #Choose files to convert to MP4
        video_file_types = (('MOV files', '*.mov'),('All files', '*.*'))
        filename = fd.askopenfilename(
                    title='Select video file(s)',
                    multiple = True,
                    filetypes = video_file_types
                    )

        #Make sure you want to continue with all the files
        res = tk.messagebox.askyesno("Video convert",f"{len(filename)} files selected. Continue?")
        if res:
            dat.bfix_index = True
            #Loop through all the files and convert them using ffmpeg
            fcount = 0
            for vfile in filename:
                fcount += 1
                if dat.bfix_index == False:
                    break
                #We make a temporary window to show the user that something is happening
                wvc = tk.Toplevel(self)
                wvc.title("Fixing broken indexing")
                wvc.geometry("400x200")
                label1 = tk.Label(wvc, text="Please wait - this may take a while")  # create the label
                label1.pack()  # add the label to the window
                label2 = tk.Label(wvc, text=f"Busy with file {fcount}/{len(filename)} ")  # create the label
                label2.pack()  # add the label to the window
                def fcancel(self,dat):
                    dat.bfix_index = False
                    wvc.destroy()
                #Place a cancel button
                btn_close = tk.Button(wvc,text = "Cancel",command = lambda:fcancel(self,dat))
                btn_close.pack()

                #Initiate the progress bar
                pbvc = ttk.Progressbar(
                 wvc,
                 orient='horizontal',
                 mode='determinate',
                 length=100
                 # value = 0
                 )
                pbvc.pack()

                #NExt we load the input file and make sure to read the creation date
                #Preserving the creation date is important for time keeping
                input_video = vfile

                #Read the video
                vid = cv2.VideoCapture(input_video)
                #Get the frame rate and count
                fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                #Get the last modified date (i.e. the saved time)
                vid_last_date = os.path.getmtime(input_video) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
                #Convert to a datetime object and subtract the total video duration (in seconds) - this gives the creation time
                creation_time = datetime.fromtimestamp(vid_last_date -(frame_count/fps))
                #Output file name
                # output_video = input_video.split(".")[0] + ".mp4"
                name, ext = os.path.splitext(vfile)
                indexed_file = f"{name}_indexed{ext}"

                #Next we use ffmpeg to fix the indexing
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-i", input_video,
                    "-c", "copy",
                    "-map", "0",
                    "-movflags", "+faststart",
                     "-metadata", f"creation_time={creation_time}",
                    indexed_file
                ]

                try:
                    #Initialise the process to run in the background
                    bg_process = subprocess.Popen(ffmpeg_command)

                    #Counters for the progressbar
                    i_wait = 0
                    b_up = True

                    #Keep moving the progressbar up and down until the process is done
                    while(bg_process.poll() is None):
                        #Update the window
                        wvc.update_idletasks()
                        self.update()

                        pbvc['value'] = i_wait * 10
                        if b_up:
                            i_wait += 1
                            if i_wait > 10:
                                b_up = False
                        else:
                            i_wait -= 1
                            if i_wait < 0:
                                b_up = True
                        time.sleep(0.05)

                    pbvc.destroy()
                    wvc.destroy()

                    print("Indexing successful.")
                except subprocess.CalledProcessError as e:
                    print("Indexing failed:", e)

        else:
            pass
    #END fix_video_index

    def convert_video(self,dat):
        #Choose files to convert to MP4
        video_file_types = (('AVI files', '*.avi'),('All files', '*.*'))
        filename = fd.askopenfilename(
                    title='Select video file(s)',
                    multiple = True,
                    filetypes = video_file_types
                    )

        #Make sure you want to continue with all the files
        res = tk.messagebox.askyesno("Video convert",f"{len(filename)} files selected. Continue?")
        if res:
            dat.bconvert = True
            #Loop through all the files and convert them using ffmpeg
            fcount = 0
            for vfile in filename:
                fcount += 1
                if dat.bconvert == False:
                    break
                #We make a temporary window to show the user that something is happening
                wvc = tk.Toplevel(self)
                wvc.title("MP4 Conversion")
                wvc.geometry("400x200")
                label1 = tk.Label(wvc, text="Please wait - this may take a while")  # create the label
                label1.pack()  # add the label to the window
                label2 = tk.Label(wvc, text=f"Busy with file {fcount}/{len(filename)} ")  # create the label
                label2.pack()  # add the label to the window
                def fcancel(self,dat):
                    dat.bconvert = False
                    wvc.destroy()
                #Place a cancel button
                btn_close = tk.Button(wvc,text = "Cancel",command = lambda:fcancel(self,dat))
                btn_close.pack()

                #Initiate the progress bar
                pbvc = ttk.Progressbar(
                 wvc,
                 orient='horizontal',
                 mode='determinate',
                 length=100
                 # value = 0
                 )
                pbvc.pack()

                #NExt we load the input file and make sure to read the creation date
                #Preserving the creation date is important for time keeping
                input_video = vfile

                #Read the video
                vid = cv2.VideoCapture(input_video)
                #Get the frame rate and count
                fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                #Get the last modified date (i.e. the saved time)
                vid_last_date = os.path.getmtime(input_video) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
                #Convert to a datetime object and subtract the total video duration (in seconds) - this gives the creation time
                creation_time = datetime.fromtimestamp(vid_last_date -(frame_count/fps))
                #Output file name
                output_video = input_video.split(".")[0] + ".mp4"
                #Next we build the ffmpeg command
                #We copy the audio and video properties so we don't loose any quality
                #The creation time is kept by specifying it in 'metadata'
                # -y makes sure we overwrite any previous videos with the same name
                #We re-encode to h.264 format with a constant frame rate
                frame_rate = 25

                ffmpeg_command = [
                    'ffmpeg',
                    '-i', input_video,
                    #'-c:v', 'copy',
                    '-vf', f'fps={frame_rate}',
                    '-c:v', 'libx264',
                    '-crf', '25',
                    '-preset','medium',
                    # '-c:a', 'aac',
                    '-f','mp4',
                    # '-r', '25',
                    '-metadata', f'creation_time={creation_time}',
                    output_video,
                    "-y"
                ]

                try:
                    #Initialise the process to run in the background
                    bg_process = subprocess.Popen(ffmpeg_command)

                    #Counters for the progressbar
                    i_wait = 0
                    b_up = True

                    #Keep moving the progressbar up and down until the process is done
                    while(bg_process.poll() is None):
                        #Update the window
                        wvc.update_idletasks()
                        self.update()

                        pbvc['value'] = i_wait * 10
                        if b_up:
                            i_wait += 1
                            if i_wait > 10:
                                b_up = False
                        else:
                            i_wait -= 1
                            if i_wait < 0:
                                b_up = True
                        time.sleep(0.05)

                    pbvc.destroy()
                    wvc.destroy()

                    print("Conversion successful.")
                except subprocess.CalledProcessError as e:
                    print("Conversion failed:", e)

        else:
            pass

class Menu_functions_PLOT(Data_Frame):
    def plot_dat(self,dat):

        # print(dat.p_cols)
        plot_perc = dat.video_width/1440
        screen_dpi =self.winfo_fpixels('1i')
        iscale = 0.95

        #Try to clear existing plot
        try:
            dat.figure_canvas_zoom.get_tk_widget().destroy()
            dat.figure_canvas_main.get_tk_widget().destroy()
        except:
            pass

        #Plot zoom figure
        dat.figure_zoom = Figure(figsize = (dat.data_frame.winfo_width()/self.winfo_fpixels('1i')*iscale,(dat.data_frame.winfo_height()*0.7)/dat.data_frame.winfo_fpixels('1i')*iscale),dpi = 100)#Zoom figure with subset according to slider
        dat.figure_canvas_zoom = FigureCanvasTkAgg(dat.figure_zoom, dat.data_frame)
        dat.ax_zoom = dat.figure_zoom.add_subplot()

        # Extracting the data for zoom plot
        #Specificy the rows to plot
        row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
        # Plot each column individually
        for col in dat.p_cols:
            # Select the data for the current column
            x_data = dat.df.iloc[row_to_use, dat.time_col]
            y_data = dat.df.iloc[row_to_use, col]

            # Exclude NA values
            non_na_mask = ~y_data.isna()
            x_data = x_data[non_na_mask]
            y_data = y_data[non_na_mask]

            # Plot the current column
            dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)

        # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns)
        dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.85, 0.9))
        dat.ax_zoom.set_xlabel('X-axis Label')
        # dat.ax_zoom.tick_params(axis='x', rotation=30)
        # dat.ax_zoom.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=7))
        # # Formatting date labels on x-axis
        # dat.ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Adjust the date format as needed

        dat.vline3 = dat.ax_zoom.axvline(x = dat.df.iloc[dat.sub_min,dat.time_col],color = 'green')
##        NavigationToolbar2Tk(dat.figure_canvas_zoom, self)

        #Attach a function to the button press event in the main plot
        dat.figure_canvas_zoom.mpl_connect("scroll_event", dat.zoom_scroll)
        dat.figure_canvas_zoom.mpl_connect("button_press_event", dat.zoom_zoom)
        dat.figure_canvas_zoom.get_tk_widget().grid(column = 0, row = 0)
        # dat.figure_canvas_zoom.get_tk_widget().pack(side=tk.BOTTOM, anchor = 'w')
        # dat.figure_canvas_zoom.get_tk_widget().place(relx = 0, rely =0.3)

        #Plot main figure
##
        # dat.figure_main = Figure(figsize = (10*plot_perc,4*plot_perc),dpi = 100) #Main figure with all data
        dat.figure_main = Figure(figsize = (dat.data_frame.winfo_width()/dat.data_frame.winfo_fpixels('1i')*iscale,(dat.data_frame.winfo_height()*0.3)/self.winfo_fpixels('1i')*iscale),dpi = 100) #Main figure with all data
        dat.figure_canvas_main = FigureCanvasTkAgg(dat.figure_main, dat.data_frame)
        dat.ax_main = dat.figure_main.add_subplot()
        # print(dat.nrow)

        i_int = round(dat.nrow/2000)
        #Specificy the rows to plot
        row_to_use = np.arange(0,(dat.nrow),i_int)
        # Plot each column individually
        for col in dat.p_cols:
            # Select the data for the current column
            x_data = dat.df.iloc[row_to_use, dat.time_col]
            y_data = dat.df.iloc[row_to_use, col]

            # Exclude NA values
            non_na_mask = ~y_data.isna()
            x_data = x_data[non_na_mask]
            y_data = y_data[non_na_mask]

            # Plot the current column
            dat.ax_main.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
        # dat.ax_main.plot(dat.df.iloc[np.arange(0,(dat.nrow),1000),dat.time_col],dat.df.iloc[np.arange(0,(dat.nrow),1000),dat.p_cols])
        dat.vline1 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_min,dat.time_col],color = 'red')
        dat.vline2 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_max,dat.time_col],color = 'red')


        #Attach a function to the button press event in the main plot
        dat.figure_canvas_main.mpl_connect("button_press_event", dat.main_zoom)
        dat.figure_canvas_main.get_tk_widget().grid(column = 0, row = 1,columnspan = 2)
        # dat.figure_canvas_main.get_tk_widget().pack(side=tk.BOTTOM, anchor = 'w')
        # # dat.figure_canvas_main.get_tk_widget().place(relx = 0,rely = 0.75)
        # dat.figure_canvas_main.get_tk_widget().place(in_ =dat.figure_canvas_zoom.get_tk_widget(), relx = 0,rely = 1)

        try:
            dat.btn_3['state'] = 'disabled' #Disable cheatsheet button
        except:
            pass
        dat.bplot = True


    def choose_axes(self,dat):
        w2 = tk.Toplevel(self)
        w2.title("Plotting axes")
        w2.geometry("800x1000")
        w2.attributes('-topmost',True)
        def fapply(self,dat):
            c = 0
            dat.p_cols = []
            for i in dat.col_names:
                x = eval("dat.c" + str(c))
                if(x.get() == 1):
                    dat.p_cols.append(c)
                    # print(dat.p_cols)
                c = c+1
##            self.plot_dat(self,dat)
            w2.destroy()

        btn_close = tk.Button(w2,text = "Apply",command = lambda:fapply(self,dat))
        btn_close.pack()
        c = 0
        ix = 0
        iy = 50

        dat.col_names = dat.df.columns
        # print(dat.col_names)
        for i in dat.col_names:
            # print(c)
            exec("dat.c" + str(c) +" = tk.IntVar()")
            exec("dat.cf" + str(c) +" = tk.IntVar()")

            #Check if the checkbox has been activated before
            temp_checkbox = tk.Checkbutton(w2, text=i,variable=eval("dat.c" + str(c)), onvalue=1, offvalue=0 )
            temp_checkbox.place(x = ix, y=iy)
            if c in dat.p_cols:
                temp_checkbox.select()


            c = c +1
            iy = iy + 100
            if iy > 1000:
                iy = 50
                ix = ix + 150

    def plot_spec(self,dat):
        # w3 = tk.Toplevel(self)
        # w3.title("Plot")
        # w3.geometry("1366x768")
        # def fclose():
        #     w3.destroy()
        # btn_close = tk.Button(w3,text = "Close",command = lambda:fclose())
        # btn_close.pack()

        t_df = dat.df.iloc[np.arange(dat.dive_min,dat.dive_max,1),:]
        x=plt.specgram(t_df["accX"], Fs = dat.frequency)
        plt.title('matplotlib.pyplot.specgram() Example\n',
                  fontsize = 14, fontweight ='bold')
        plt.show()
    def notch_filter(dat):
        t_df = dat.df.iloc[np.arange(dat.dive_min,dat.dive_max,1),:]
        # Create subplot
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)

        fs = dat.frequency  # Sample frequency (Hz)
        f0 = 4  # Frequency to be removed from signal (Hz)
        Q = 10  # Quality factor
        # Design notch filter
        b, a = signal.iirnotch(f0, Q, fs)
        y_notched = signal.filtfilt(b, a, t_df["accZ"])

        l, = plt.plot(y_notched)

        # Create axes for frequency and amplitude sliders
        axfreq = plt.axes([0.25, 0.15, 0.65, 0.03])
        axremove = plt.axes([0.25, 0.1, 0.65, 0.03])
        axq = plt.axes([0.25, 0.05, 0.65, 0.03])
        # Create a slider from 0.0 to 20.0 in axes axfreq
        # with 3 as initial value
        freq_base = Slider(axfreq, 'Base Frequency', 1, 100, dat.frequency)

        # Create a slider from 0.0 to 10.0 in axes axfreq
        # with 5 as initial value and valsteps of 1.0
        freq_remove = Slider(axremove, 'Frequency to remove', 1,
                            fs, 4, valstep=0.1)

        qs = Slider(axq, 'Q', 0,
                            30, 10, valstep=0.1)


        # Create fuction to be called when slider value is changed

        def update(val):
            f = int(freq_base.val)
            fr = freq_remove.val
            q = qs.val
            b, a = signal.iirnotch(fr, q, f)
            y_notched = signal.filtfilt(b, a, t_df["accZ"])
            l.set_ydata(y_notched)

            # sos = signal.butter(f, a, 'lp', fs=1000, output='sos')
            # filtered = signal.sosfiltfilt(sos, df["x"])
            # l.set_ydata(filtered)
            # l.set_ydata(a*np.sin(2*np.pi*f*t))

        # Call update function when slider value is changed
        freq_base.on_changed(update)
        freq_remove.on_changed(update)
        qs.on_changed(update)

        # display graph
        plt.show()

    def plot_audio(self,dat):
        plot_perc = dat.video_width/1440

        #Try to clear existing plot
        try:
            dat.figure_canvas_audio.get_tk_widget().destroy()
        except:
            pass

        #Plot zoom figure
        #Update Audio plot
        if dat.audio_present == True:
            print("Plotting audio")
            # print("len aud: " + str(len(dat.audio)))
            dat.sub_window = dat.sub_max - dat.sub_min
            # print("sub_window: " + str(dat.sub_window))
            dat.aud_window = int((dat.sub_window*640)/2)
            # print("aud_window: " + str(dat.aud_window))

            dat.aud_point = int(dat.frame*640)
            # print("frame: " + str(dat.frame))
            # print("aud_point: " + str(dat.aud_point))
            if dat.frame > 1:
                dat.aud_min = dat.aud_point - dat.aud_window
            else:
                dat.aud_min = dat.aud_point
            dat.aud_max = dat.aud_point + dat.aud_window
            #Make sure the window does not stretch past the available data
            if(dat.aud_max > len(dat.audio)):
                dat.aud_max = (len(dat.audio) -1)

            # print("aud_min: " + str(dat.aud_min))
            # print("aud_max: " + str(dat.aud_max))
            dat.figure_audio = Figure(figsize = (6*plot_perc,4*plot_perc),dpi = 100)#Zoom figure with subset according to slider
            dat.figure_canvas_audio = FigureCanvasTkAgg(dat.figure_audio, self)
            dat.ax_audio = dat.figure_audio.add_subplot()
            dat.ax_audio.plot(dat.audio.Timestamp[np.arange(dat.aud_min,dat.aud_max,10)],dat.audio.aud[np.arange(dat.aud_min,dat.aud_max,10)])
            # dat.ax_audio.plot(dat.audio.Timestamp[dat.aud_min:dat.aud_max],dat.audio.aud[dat.aud_min:dat.aud_max])
            dat.ax_audio.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))

            # dat.vline3 = dat.ax_audio.axvline(x = dat.df.iloc[dat.aud_min,dat.time_col],color = 'green')
    ##        NavigationToolbar2Tk(dat.figure_canvas_zoom, self)

            #Attach a function to the button press event in the main plot
            dat.figure_canvas_audio.mpl_connect("scroll_event", dat.audio_scroll)
            dat.figure_canvas_audio.mpl_connect("button_press_event", dat.audio_zoom)

            # dat.figure_canvas_audio.get_tk_widget().pack(side=tk.BOTTOM, anchor = 'w')
            dat.figure_canvas_audio.get_tk_widget().place(in_ = dat.figure_canvas_zoom.get_tk_widget(),relx = 1, rely =0)
        else:
            print("No audio files")


class Menu_functions_ANALYSIS(Data_Frame):
    def acc_metrics(self,dat):
        #Choose the columns with the accelerometer data
        wtemp = tk.Toplevel(self)
        wtemp.title("Please select the Accelerometer columns")
        wtemp.geometry("600x400")
        wtemp.attributes('-topmost',True)

        #ACC XYZ column selection
        dat.ax_col = tk.StringVar()
        dat.ax_col .set('Select accX column')
        dat.ay_col = tk.StringVar()
        dat.ay_col .set('Select accY column')
        dat.az_col = tk.StringVar()
        dat.az_col .set('Select accZ column')

        # Column name options
        column_names = dat.df.columns.tolist()

        #accX
        ax_menu = tk.OptionMenu(wtemp, dat.ax_col, *column_names)
        tk.Label(wtemp, text="Choose accX column").grid(row=0, column=0)
        ax_menu.grid(row=0, column =1)

        #accY
        ay_menu = tk.OptionMenu(wtemp, dat.ay_col, *column_names)
        tk.Label(wtemp, text="Choose accY column").grid(row=1, column=0)
        ay_menu.grid(row=1, column =1)

        #accX
        az_menu = tk.OptionMenu(wtemp, dat.az_col, *column_names)
        tk.Label(wtemp, text="Choose accZ column").grid(row=2, column=0)
        az_menu.grid(row=2, column =1)

        #Function when the "Apply button is pressed"
        #Once the column name and format has been chosen we convert the column to POSIX
        def tapply():
            s_ax = dat.ax_col.get()
            s_ay = dat.ay_col.get()
            s_az = dat.az_col.get()
            #Fix broken/sticky ACC data
            #Check the acc axes
            #First make sure they are numeric
            dat.df[s_ax] = pd.to_numeric(dat.df[s_ax], errors='coerce')
            dat.df[s_ay] = pd.to_numeric(dat.df[s_ay], errors='coerce')
            dat.df[s_az] = pd.to_numeric(dat.df[s_az], errors='coerce')
            dat.df[s_ax] = dat.df[s_ax].interpolate()
            dat.df[s_ay] = dat.df[s_ay].interpolate()
            dat.df[s_az] = dat.df[s_az].interpolate()
            dat.df['accZ2'] = dat.df['accY2'] = dat.df['accX2'] = 9999
            dat.df['accX2'][1:] = dat.df[s_ax][:-1]
            dat.df['accY2'][1:] = dat.df[s_ay][:-1]
            dat.df['accZ2'][1:] = dat.df[s_az][:-1]
            dat.df['accXdiff'] = dat.df[s_ax] - dat.df['accX2']
            dat.df['accYdiff'] = dat.df[s_ay] - dat.df['accY2']
            dat.df['accZdiff'] = dat.df[s_az] - dat.df['accZ2']

            dat.df.loc[np.round(dat.df['accXdiff'], 3) == 0, s_ax] = np.nan
            dat.df.loc[np.round(dat.df['accYdiff'], 3) == 0, s_ay] = np.nan
            dat.df.loc[np.round(dat.df['accZdiff'], 3) == 0, s_az] = np.nan
            # na_rows = np.sum(x[s_ax].isna() | x[s_ay].isna() | x[s_az].isna())

            dat.df[s_ax] = dat.df[s_ax].interpolate()
            dat.df[s_ay] = dat.df[s_ay].interpolate()
            dat.df[s_az] = dat.df[s_az].interpolate()

            #Find the sampling rate
            date_diffs = dat.df[dat.time_col_string].diff()
            sampling_rate = int(round(1/(date_diffs).mean().total_seconds(),0))
            print("Sampling rate is: " + str(sampling_rate))
            rolling_window1 = sampling_rate
            rolling_window2 = 8


            #Rolling mean of acc axes with chosen window
            dat.df["ax_s"] = dat.df[s_ax].rolling(rolling_window1,center = True).mean()
            dat.df["ay_s"] = dat.df[s_ay].rolling(rolling_window1,center = True).mean()
            dat.df["az_s"] = dat.df[s_az].rolling(rolling_window1,center = True).mean()

            dat.df["ax_std"] = dat.df[s_ax].rolling(rolling_window2,center = True).std()
            dat.df["ay_std"] = dat.df[s_ay].rolling(rolling_window2,center = True).std()
            dat.df["az_std"] = dat.df[s_az].rolling(rolling_window2,center = True).std()

            # print(dat.df.columns)

            #Normalize acc axes
            acc_min = min(dat.df.iloc[:,np.arange((len(dat.df.columns)-3),(len(dat.df.columns)),1)].min())
            # print("acc_min: " + str(acc_min))
            acc_max = max(dat.df.iloc[:,np.arange((len(dat.df.columns)-3),(len(dat.df.columns)),1)].max())
            # print("acc_max: " + str(acc_max))
            dat.df["ax_NORM"] = 2*(dat.df.loc[:,s_ax] - acc_min)/(acc_max-acc_min)*1 -1
            dat.df["ay_NORM"] = 2*(dat.df.loc[:,s_ay] - acc_min)/(acc_max-acc_min)*1 -1
            dat.df["az_NORM"] = 2*(dat.df.loc[:,s_az] - acc_min)/(acc_max-acc_min)*1 -1

            #Dynamic acceleration
            dat.df["ax_d"] = dat.df[s_ax] - dat.df["ax_s"]
            dat.df["ay_d"] = dat.df[s_ay] - dat.df["ay_s"]
            dat.df["az_d"] = dat.df[s_az] - dat.df["az_s"]

            #Metrics calculation
            dat.df["pitch"] = np.arctan2(dat.df.loc[:,"ax_s"], np.sqrt(dat.df.loc[:,"ay_s"]**2 + dat.df.loc[:,"az_s"]**2))#*180/np.pi
            dat.df["roll"] = np.arctan2(dat.df.loc[:,"ay_s"], np.sqrt(dat.df.loc[:,"ax_s"]**2 + dat.df.loc[:,"az_s"]**2))#*180/np.pi
            dat.df["vedba"] = np.sqrt(dat.df["ax_d"]**2 + dat.df["ay_d"]**2 + dat.df["az_d"]**2)
            dat.df["vedba_std"] = dat.df['vedba'].rolling(rolling_window2,center = True).std()

            #Rolling SD of pitch (for foraging dive estimation) over 2 seconds
            dat.df["pitch_rm"] = dat.df['pitch'].rolling((rolling_window1*2), center = True).std()

            wtemp.destroy()

        #Create the button and add the
        btnApply = tk.Button(wtemp,text = "Apply",command = tapply)
        btnApply.grid(row = 3, column = 0)

    def find_dives(self,dat):
        #First specify the column to use
        #Create a new promt window to select the data column from
         wtemp = tk.Toplevel(self)
         wtemp.title("Please select the column with depth (in m)")
         wtemp.geometry("600x400")
         wtemp.attributes('-topmost',True)

        #DateTime column selection
         dat.depth_col = tk.StringVar()
         dat.depth_col.set('Select DateTime column')
         print((dat.depth_col.get()))
        # Column name options
         column_names = dat.df.columns.tolist()
         column_menu = tk.OptionMenu(wtemp, dat.depth_col, *column_names)
         tk.Label(wtemp, text="Choose a column").grid(row=0, column=0)
         column_menu.grid(row=1, column =0)

        #Function when the "Apply button is pressed"
        #Once the column name and format has been chosen we convert the column to POSIX
         def tapply():
            #Select the column from a dropdown menu
            # print(dat.depth_col.get())
            s_col = dat.depth_col.get()

            # print(dat.df.loc[0,s_col])

            #Save the column string
            dat.depth_col_string = s_col
            # #Save the column number of Timestamp
            dat.depth_col = dat.df.columns.get_loc(s_col)
            dat.temp_done = True
            wtemp.destroy()

        #Create the button and add the
         btnApply = tk.Button(wtemp,text = "Apply",command = tapply)
         btnApply.grid(row = 2, column = 0)

         #Wait for a decision on Depth col
         dat.temp_done = False
         while dat.temp_done == False:
             self.update()


        #Automatic detection of dives
         print("Finding dives...")

         if('DIVE' in dat.df):
             dat.dive_num = dat.df["DIVE"].max()
         else:
             dat.df['DIVE'] = 0
             dat.dive_num = 0

        #interpolate the values to match the ACC data
         dat.df.loc[:,dat.depth_col_string] =  dat.df.loc[:,dat.depth_col_string].interpolate()
         #Normalize for easier plotting
         dat.df["Depth_NORM"] = (dat.df.loc[:,dat.depth_col_string] - np.nanmin(dat.df.loc[:,dat.depth_col_string]))/(np.nanmax(dat.df.loc[:,dat.depth_col_string])-np.nanmin(dat.df.loc[:,dat.depth_col_string]))*2


         #Check if the dives have been processed before
         pre,ext = os.path.splitext(dat.filename)
         out_file = pre + "_DIVES.csv"

         if os.path.exists(out_file):
             print("Dives already done")
             dives = pd.read_csv(out_file)
             dat.df = dat.df.join(dives)
            #Normalize acc data

             # print(dat.df.columns)
             # res = tk.messagebox.askyesno("Find Dives","The video has already been squashed. Are you sure you want to do it again?")
             # if res:
             #     dat.bsquash = True
             # else:
             #     dat.bsquash = False
         else:
             dat.bdives = True #Set the flag to show dive analysis is in progress.
             dat.df = dat.df.assign(dive = 0)
             dat.df = dat.df.assign(forage_dive = 0)
             dat.df.loc[round(dat.df['Depth'], 2) >= 0.4, 'dive'] = 1

             #Find changepoints for dives
             dat.df = dat.df.assign(dpoint = 0)
             dat.df['dpoint'].iloc[0:(dat.df.shape[0]-1)] = np.diff(dat.df['dive'])
             dpoints = np.where(dat.df['dpoint'] != 0)[0] #Points where depth crosses the threshold line
             dive_ss = dat.df.iloc[dpoints,:] #rows in the df corresponding to changepoints

           #We are assuming that the tag starts above the surface, so if the first data are below the threshold, we assume it is an error, and discard
             if dive_ss.iloc[0]['dpoint'] == -1:
               dat.df.loc[dpoints[0],'dpoint'] = 0 #replace this change point with a zero as we don't want to start the data in a dive (it is impossible)
               dpoints = np.where(dat.df['dpoint'] != 0)[0] #redo the step from above
               dive_ss = dat.df.iloc[dpoints,:] #rows in the df corresponding to changepoints

             #Get time between changepoints
             dive_ss = dive_ss.assign(dss = 0)
             #Get the difference between the dive end time and dive start time
             #First for end points
             dive_ss['dss'].iloc[range(1, dive_ss.shape[0], 2)] = (dive_ss[dat.time_col_string].iloc[range(1, dive_ss.shape[0], 2)].values - dive_ss[dat.time_col_string].iloc[range(0, (dive_ss.shape[0]-1), 2)].values).astype('timedelta64[s]').astype(float)
             #This is repeated for start points
             dive_ss['dss'].iloc[range(0, (dive_ss.shape[0]-1), 2)] = (dive_ss[dat.time_col_string].iloc[range(1, dive_ss.shape[0], 2)].values - dive_ss[dat.time_col_string].iloc[range(0, (dive_ss.shape[0]-1), 2)].values).astype('timedelta64[s]').astype(float)
             dives = dpoints#[np.where(dive_ss['dss'] > duration)]
             # ggplotly(g+geom_point(data = dive_ss[np.where(dive_ss['dss'] > duration),:],aes(Timestamp,Depth),col = 2)+geom_line(aes(Timestamp,1)))


             #Assign dive changepoints to the data
             dat.df = dat.df.assign(dpoint = 0)
             dat.df.loc[dives, 'dpoint'] = dive_ss['dpoint']#[np.where(dive_ss['dss'] > duration)]


             # id = dat.df['TagID'][0]

             #Get dive numbers
             dat.df = dat.df.assign(dive_num = np.nan,
                                first_pce_pitch= np.nan,
                                dive_max_depth= np.nan,
                                dive_dur= np.nan,
                                # dive_tot_pce= np.nan,
                                dive_max_temp= np.nan,
                                dive_behav= np.nan,
                                dive_behav2= np.nan,
                                dive_pitch_sd = np.nan,
                                forage = "normal")

             dive_num = 0

             # length(which(data$dpoint ==1))
             # print(len(dat.df[dat.df["dpoint"] == 1]))


             #Create a new window to show a progress bar and cancel button
             #New temp window
             w2 = tk.Toplevel(self)
             w2.title("Finding dives")
             w2.geometry("400x200")
             label = tk.Label(w2, text="Please wait - this may take a while")  # create the label
             label.pack()  # add the label to the window
             def fcancel(self,dat):
                 dat.bdives = False
                 w2.destroy()
             #Place a cancel button
             btn_close = tk.Button(w2,text = "Cancel",command = lambda:fcancel(self,dat))
             btn_close.pack()

             #Initiate the progress bar
             pb = ttk.Progressbar(
              w2,
              orient='horizontal',
              mode='determinate',
              length=100
              # value = 0
              )
             pb.pack()
             # #Loop through all the frames and save the first column (pixels) to the empty image
             print("Finding dives...")

             icounter = 0
             for ipt in np.where(dat.df["dpoint"] == 1)[0]:
                 if dat.bdives == True:
                     icounter = icounter + 1
                     dive_proportion = round(icounter/len(np.where(dat.df["dpoint"] == 1)[0]),2)*100
                     #Update progress bar
                     pb['value'] = dive_proportion
                     w2.update_idletasks()
                     self.update()

                     #Increment the dive number
                     dive_num += 1

                     #Starting row and Timestamp of the dive
                     rstart = ipt
                     dive_start_time = dat.df.at[rstart, dat.time_col_string]

                     #End row and Timestamp of the dive
                     next_dive_ends = np.where(dat.df["dpoint"][ipt:] == -1)[0]
                     #Make sure there is a end to the dive within the df
                     if len(next_dive_ends) > 0:
                       #Get row number of dive end
                       rend = ipt + next_dive_ends[0]

                       # tnow <- Sys.time()
                       temp_data = dat.df.loc[rstart:rend].reset_index()

                       #Reset the dive column and forage column
                       temp_data.dive = 0
                       temp_data.forage_dive = 0

                       #Timestamp
                       dive_end_time = dat.df.at[rend, dat.time_col_string]
                       #Duration of the dive (in seconds)
                       dive_dur = (dive_end_time - dive_start_time).total_seconds()

                       #Maximum depth of the dive
                       dive_max_depth = temp_data[dat.depth_col_string].max()

                       #Maximum temperature of dive
                       dive_max_temp = "NA"#temp_data["Temp"].max()

                       #Total PCE in dive (only looking at PCE == 1)

                       # tot_pce = temp_data.loc[temp_data["PCE"] <= 1, "PCE"].sum()

                       #Find the SD pitch of the dive (this indicates foraging when high - i.e. > 15 degrees)
                       if "pitch" in temp_data.columns:
                           dive_pitch_sd = (temp_data.loc[:, "pitch"]*180/np.pi).std()
                       else:
                           dive_pitch_sd = "NA"

                       #Write the corresponding data to the df
                       temp_data = temp_data.assign(dive_num=dive_num,
                                                    dive_pitch_sd = dive_pitch_sd,
                                                    dive_dur = dive_dur,
                                                    dive_max_depth = dive_max_depth,
                                                    dive_max_temp = dive_max_temp,
                                                    # dive_tot_pce = tot_pce,
                                                    d_rate = np.nan)

                       #Behaviour of dive (foraging or not)
                       #Do this only for dives > 3m (see Beck et al. 2000, Luque et al. 2008)
                       #Beck CA, Bowen WD, Iverson SJ (2000) Seasonal changes in buoyancy and diving behaviour of adult grey seals. Journal of Experimental Biology 203:2323-2330
                       #Luque SP, Arnould JPY, Guinet C (2008) Temporal structure of diving behaviour in sympatric Antarctic and subantarctic fur seals. Marine Ecology Progress Series 372:277-287

                       if dive_max_depth >= 3:

                         #Mark as dive only if deep and long enough
                         if dive_dur >= 20:
                             temp_data.dive = 1
                             #Mark as forage dive if pitch SD is
                             # if dive_pitch_sd >= 15:
                             #     temp_data.forage_dive = 1

                         temp_data = temp_data.assign(d_depth = 0,
                                                       dt = 0)

                         #Calculate the rate of change for dives
                         for i in range(len(temp_data)-1):
                             temp_data.loc[i, "d_depth"] = temp_data.at[i+1, dat.depth_col_string] - temp_data.at[i, dat.depth_col_string]
                             temp_data.at[i, "dt"] = (temp_data.at[i+1, dat.time_col_string] - temp_data.at[i, dat.time_col_string]).total_seconds()
                         temp_data["d_rate"] = temp_data["d_depth"]/temp_data["dt"]

                         def f_scale(val,lower=0,upper=1):
                             val_out = ((upper-lower)*(val-min(val))/(max(val) - min(val)) )+ lower
                             return(val_out)
                         #Scale the rate of change
                         temp_data["d_rate"] = f_scale(temp_data["d_rate"],-1,1)

                         #Scale the depth
                         temp_data["depth_scale"] = f_scale(temp_data[dat.depth_col_string], 0,1)

                         #Find phases
                         #bottom phase is where Depth is > %50 of max depth and rate of change is between -0.5 and 0.5
                         try:
                             bottom_start = temp_data.loc[(temp_data['d_rate'] <= 0.5) & (temp_data['depth_scale'] > 0.5)].index[0]
                         except:
                             bottom_start = temp_data.loc[temp_data['depth_scale'] > 0.5].index[0]
                         try:
                             bottom_end = temp_data.loc[(temp_data['d_rate'] <= -0.5) & (temp_data['depth_scale'] < 0.5)].index[0]
                         except:
                             bottom_end = temp_data.loc[(temp_data['depth_scale'] < 0.5) & (temp_data[dat.time_col_string] > temp_data.iloc[bottom_start][dat.time_col_string])].index[0]

                         # plt.plot(temp_data[dat.depth_col_string])
                         # plt.plot(temp_data["depth_scale"])
                         # plt.plot(temp_data["d_rate"])

                         temp_data['dive_behav'][0:bottom_start] = 'descent'
                         temp_data['dive_behav'][bottom_start:bottom_end] = 'bottom'
                         temp_data['dive_behav'][bottom_end:len(temp_data)] = 'ascent'
                         temp_data['dive_behav2'] = temp_data['dive_behav'] + '_' + temp_data['forage']


                       #Check if foraging has occurred
                       # if tot_pce > 0:
                       #
                       #   # break
                       #   #Get the smoothed pitch maximum before the foraging starts
                       #   pce_indices = np.where(temp_data['PCE'] == 1)[0]
                       #   first_pce_pitch = temp_data['pitch_rm'][0:pce_indices[0]+1].max()
                       #   temp_data['first_pce_pitch'] = first_pce_pitch
                       #
                       #
                       #   #Get the foraging events
                       #   pces = pce_indices
                       #   forage = np.full(len(temp_data), '', dtype='object')
                       #   forage[pces[0]:pces[-1]+1] = 'forage'
                       #   temp_data['forage'] = forage
                       #
                       #   # Update dive_behav2
                       #   dive_behav2 = temp_data['dive_behav'].values.copy()
                       #   for i in range(pces[0], pces[-1]+1):
                       #       dive_behav2[i] = f"{temp_data['dive_behav'][i]}_{temp_data['forage'][i]}_{round(tot_pce, -1)}"
                       #   temp_data['dive_behav2'] = dive_behav2
                       #   # Find unique dive_behav values
                       #   np.unique(temp_data['dive_behav'])


                       # temp_data['dive_behav'] = temp_data['dive_behav2']

                       #Select columns to export
                       col_select = ["Depth_NORM","dive","forage_dive", "dpoint","d_rate", "dive_num", "dive_pitch_sd","first_pce_pitch", "dive_max_depth", "dive_dur", "dive_max_temp", "forage", "dive_behav", "dive_behav2"]
                       dat.df.loc[rstart:rend, col_select] = temp_data.loc[:, col_select].values


         # pce_out = self.df[["PCE","vid","frame","vid_time","BEHAV"]]

         # pce_out.to_csv(out_file,index = False)

             if dat.bdives == True:
                 #Get the accelerometer column names
                 #s_ax = dat.ax_col.get()
                 #s_ay = dat.ay_col.get()
                 #s_az = dat.az_col.get()
                 #Normalize acc values according to dives
                 #max_X = np.max(dat.df.accX[dat.df.dive_max_depth >= 3])
                 #min_X = np.min(dat.df.accX[dat.df.dive_max_depth >= 3])
                 #max_Y = np.max(dat.df.accY[dat.df.dive_max_depth >= 3])
                 #min_Y = np.min(dat.df.accY[dat.df.dive_max_depth >= 3])
                 #max_Z = np.max(dat.df.accZ[dat.df.dive_max_depth >= 3])
                 #min_Z = np.min(dat.df.accZ[dat.df.dive_max_depth >= 3])

                 #dat.df['accX_norm_dive'] = -1 + (dat.df.accX - min_X) * (1 - (-1)) / (max_X - min_X)
                 #dat.df['accY_norm_dive'] = -1 + (dat.df.accY - min_Y) * (1 - (-1)) / (max_Y - min_Y)
                 #dat.df['accZ_norm_dive'] = -1 + (dat.df.accZ - min_Z) * (1 - (-1)) / (max_Z - min_Z)

                 dat.df.loc[~np.isnan(dat.df['dive_num']), 'dive_id'] = dat.df.loc[~np.isnan(dat.df['dive_num']), 'TagID'] + '_' + dat.df.loc[~np.isnan(dat.df['dive_num']), 'dive_num'].astype(int).astype(str)
                 #col_select = ["dive","dive_id","accX_norm_dive","accY_norm_dive","accZ_norm_dive","forage_dive", "dpoint","d_rate", "dive_num", "dive_pitch_sd","first_pce_pitch", "dive_max_depth", "dive_dur",  "dive_max_temp", "forage", "dive_behav", "dive_behav2"]
                 col_select = ["dive","dive_id","forage_dive", "dpoint","d_rate", "dive_num", "dive_pitch_sd","first_pce_pitch", "dive_max_depth", "dive_dur",  "dive_max_temp", "forage", "dive_behav", "dive_behav2"]


                 if dat.view_only == False:
                     dat.df.loc[:,col_select].to_csv(out_file,index = False)
                     print("Dives SAVED")
                 else:
                     print("view only mode file not saved")
                 #Print the total number of foraging dives
                 # f_dives = dat.df.loc[dat.df["dive_pitch_sd"] > 15 ,:]
                 # n_f_dives = len(np.unique(f_dives['dive_num']))
                 # print("Total foraging dives: " + str(n_f_dives))

                 dat.dives_loaded = True

                 dat.bdives = False #Reset the flag
                 w2.destroy()
    def squash_vid(self,dat):
        print("Squash vid")
        dat.bsquash = True
        # #Choose a video file to be squashed
        # file_types = (('avi files', '*.mp4'),('All files', '*.*'))
        # filename = fd.askopenfilename(
        #             title='Choose a file',
        #             initialdir=dat.wd,
        #             filetypes=file_types)
        # # filename = "C:/Users/SStho/Documents/Temp_data/3/2022_01_07_AC2104/AC2104_HPM03 (00).mp4" #DEBUGGING
        v_num = -1
        files = sorted(os.listdir(dat.wd))
        # print(files)
        # for v_num in range(len(files)):
        while (v_num < len(files)) and dat.bsquash == True:
            # if(v_num < len(files)):
                # filename = dat.wd +'/' + files[v_num]
                # print(files[v_num])
                v_num = v_num + 1
                # print(files[v_num])
                while(((not files[v_num].lower().endswith(".mp4")))  and v_num < len(files)):
                    v_num = v_num + 1
                # print(v_num)
                filename = dat.wd +'/' + files[v_num]
                # print(files[v_num])
                #Check if the chosen video has already been squashed (it should have a "_squash.jpg" suffix)
                squash_file = filename.split(".")[0] + "_squash.jpg"
                # print(squash_file)
                dat.bsquash_present = True #monitor if squashing should continue
                if os.path.exists(squash_file):
                    dat.bsquash_present = False
                    # res = tk.messagebox.askyesno("Video squash","The video has already been squashed. Are you sure you want to do it again?")
                    # if res:
                    #     dat.bsquash = True
                    # else:
                    #     dat.bsquash = False

                #Squash image
                if dat.bsquash_present == True and dat.bsquash == True:
                   #Load the video
                   cap = cv2.VideoCapture(filename)
                   #Get the number of frames - this will be the width of the output image
                   frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                   fps = int(cap.get(cv2.CAP_PROP_FPS))
                   frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                   v_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                   v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                   #Horizontal image
                   #Keep rows (height) and replace cols (width) with video frames
                   # img = np.zeros((v_height, frame_count+1, 3), np.uint8)

                   #Vertical image
                   #Keep cols (width) and replace rows (height) with video frames
                   img = np.zeros((frame_count+1, v_width, 3), np.uint8)

                   #Choose the reference row
                   ref_row = int(v_height*0.3)

                   #Create a new window to show a progress bar and cancel button
                   #New temp window
                   w2 = tk.Toplevel(self)
                   w2.title("Squashing image")
                   w2.geometry("400x200")
                   label = tk.Label(w2, text="Please wait - this may take a while")  # create the label
                   label.pack()  # add the label to the window
                   def fcancel(self,dat):
                       print("Squashing cancelled")
                       dat.bsquash = False
                       w2.destroy()
                   #Place a cancel button
                   btn_close = tk.Button(w2,text = "Cancel",command = lambda:fcancel(self,dat))
                   btn_close.pack()

                   #Initiate the progress bar
                   pb = ttk.Progressbar(
                    w2,
                    orient='horizontal',
                    mode='determinate',
                    length=350
                    # value = 0
                    )
                   pb.pack()
                   # #Loop through all the frames and save the first column (pixels) to the empty image
                   print("Squashing in progress...")
                   while True:
                        # Read a frame from the video
                        ret, frame = cap.read()

                        frame_proportion = int(frame_num/frame_count*100)
                        if ret:
                            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            # print(frame_num)
                            # print(frame_proportion)
                            pb['value'] = frame_proportion
                            w2.update_idletasks()
                            self.update()

                            #Extract a single row/col from the frame
                            # frame2 = frame[:,1,:] #Horizontal image
                            frame2 = frame[ref_row,:,:].copy() #vertical image

                            #Get the mean value for each channel
                            frame2[:,0] = frame2[:,0].mean()
                            frame2[:,1] = frame2[:,1].mean()
                            frame2[:,2] = frame2[:,2].mean()

                            # img[:,frame_num,:] = frame2  #Horizontal image
                            img[frame_num,:,:] = frame2  #Vertical image
                        else:
                            frame_num = frame_num + 1

                            # img[:,frame_num,:] = 0  #Horizontal image
                            img[frame_num,:,:] = 0  #Vertical image
                            print(f"Frame {frame_num} missing")

                            # print(f"Next frame is  {frame_num} ")
                            # print(f"Out of: {frame_count} ")
                            cap.set(1,frame_num)


                        # If there are no more frames, break out of the loop
                        if (frame_num >= frame_count)  or (dat.bsquash == False):
                            print("end of video")
                            break


                    #Smooth image pixels over 1 second
                   print("smoothing image")
                   img_smooth = img.copy()
                   # print(range(v_width)) #DEBUG
                   for row in range(v_width):
                        # print(row) #DEBUG
                        img0 = pd.DataFrame(img[:,row,0]).rolling(fps).max()
                        img1 = pd.DataFrame(img[:,row,1]).rolling(fps).max()
                        img2 = pd.DataFrame(img[:,row,2]).rolling(fps).max()
                        img_smooth[:,row,0] = img0.iloc[:,0]
                        img_smooth[:,row,1] = img1.iloc[:,0]
                        img_smooth[:,row,2] = img2.iloc[:,0]

                   # #Look for dive starts
                   # gray = cv2.cvtColor(img_smooth,cv2.COLOR_BGR2GRAY)
                   # blur = cv2.GaussianBlur(gray,(11,11),0)                          #blur image to reduce noise
                   #
                   # #Apply threshold to form a binary image
                   # ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #transform image to binary image
                   #
                   # ts_diff = pd.DataFrame(np.diff(th4[:,0]),columns = ["val"]) #1 is from white to black - 255 is from black to white
                   # ts_diff['frame'] = range(len(ts_diff))
                   # ts_diff['dive'] = 0
                   # ddur = fps*5
                   #
                   # th4 =cv2.rotate(th4, cv2.ROTATE_90_COUNTERCLOCKWISE)
                   # cv2.imwrite(filename.split(".")[0] + "_squash_TH.jpg", th4)
                   #
                   # for row in range(len(ts_diff)):
                   #     if row < ddur:
                   #         next
                   #     if ts_diff['val'][row] == 1:
                   #         # print(row)
                   #         # break
                   #         if (sum(ts_diff.loc[(row-ddur):(row-1),'val']) == 0) and (sum(ts_diff.loc[(row+1):(row+ddur),'val']) == 0):
                   #             # print("######################")
                   #             print(row)
                   #             ts_diff.loc[row,'dive'] = 1
                   #
                   # #Draw a line at the points where the dives are identified
                   # img[ts_diff[ts_diff['dive'] == 1].index,:,:] = [0,0,255]
                   #

                   #For vertical image
                   # img =cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                   img_rot =cv2.rotate(img_smooth, cv2.ROTATE_90_COUNTERCLOCKWISE)
                   print("Saving image - please wait")
                   if dat.bsquash == True:
                       cv2.imwrite(squash_file, img_rot)
                       w2.destroy()
                       # showinfo(title='Image squashed',message=squash_file)
                       print("Squashed image saved")

    def sync_dives_manual(self,dat):
        # if dat.video_loaded == False or dat.df_loaded == False or dat.dives_loaded == False:
        #     showinfo(title='Dive time synchronize',message= "Please load DATA and VIDEO first and identify DIVES")
        # else:
            w2 = tk.Toplevel(self)
            w2.title("Dive time synchronize")
            w2.geometry("800x1000")
            def fsynch(self,dat):
                #Find the dive times by subtracting the end frame number from the start
                #Divide by FPS to get back to seconds
                d1diff = np.round((dat.d1b - dat.d1a)/dat.fps)
                # print(d1diff)
                d2diff = np.round((dat.d2b - dat.d2a)/dat.fps)
                # print(d2diff)
                d3diff = np.round((dat.d3b - dat.d3a)/dat.fps)
                # print(d3diff)
                #Save three times to an array
                vid_diff = pd.DataFrame([d1diff,d2diff,d3diff],columns = ["diff"])
                # print(vid_diff)

                #Try to find these three dives in the TDR data
                dive = dat.df[dat.df['dive_dur'] > 20]
                dive = dive[dive['dive_max_depth'] > 3]

                udives = np.unique(dive['dive_num']) #Find unique dive numbers
                #Loop through the dives and match consecutive dives with with the above vid times
                mdiff_min = 9999 #A holder for the best fit
                for d in range((len(udives)-len(vid_diff))):
                    d1dur = np.round(dive.loc[dive['dive_num'] == udives[d],'dive_dur']).reset_index()
                    d2dur = np.round(dive.loc[dive['dive_num'] == udives[d+1],'dive_dur']).reset_index()
                    d3dur = np.round(dive.loc[dive['dive_num'] == udives[d+2],'dive_dur']).reset_index()


                    tdrdiff = pd.DataFrame([d1dur.loc[0,'dive_dur'],d2dur.loc[0,'dive_dur'],d3dur.loc[0,'dive_dur']],columns=["diff"])
                    # print(tdrdiff)

                    mdiff = np.mean(np.round(np.abs(tdrdiff['diff'].values - vid_diff['diff'].values)))
                    # print(mdiff)
                    if mdiff < mdiff_min:
                        mdiff_min = mdiff
                        d_hold = d
                        # print("#######################")
                        # print(vid_diff)
                        # print(tdrdiff)
                        # print(mdiff)
                        # print(d)
                        # print(udives[d])

                #Take the the value with the best fit
                #But reject the result if the time difference is too large
                if mdiff_min > 3:
                    showinfo("Dive sync ERROR","Can't synch, time difference is too large")
                    # print(mdiff_min)
                    pass
                #Synch the data
                else:
                    showinfo("Dive sync SUCCESFUL","Time synchronized, please check to make sure")
                    #point the IMU to the right point
                    d1time = dive.loc[dive['dive_num'] == udives[d_hold],dat.time_col_string].reset_index()
                    dat.IMU_date = d1time.loc[0,dat.time_col_string]
                    # print(dat.IMU_date)

                    #point the vid to the right point
                    dat.frame = dat.d1a
                    dat.vid.set(1,dat.frame)
                    Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()
                    #Synchronise the time
                    dat.date_set = 1
                    print('synch video and IMU')
                    # print(dat.IMU_date)
                    # print(dat.vid_start_date)

                    dur = dat.frame_count/dat.fps

                    idx_start1 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
                    idx_end1 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick

                    pce_hold = dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc(dat.current_label_col)]
                    dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc(dat.current_label_col)] = 0

                    offset_hold = dat.video_offset
                    # print("holding offset: " + str(offset_hold))
                    dat.video_offset = dat.IMU_date - dat.frame_date
                    # print("new offset: " + str(dat.video_offset))
                    vid_start_date = dat.vid_start_date + dat.video_offset
                    dat.vid_start_date = vid_start_date
                    dat.video_offset = offset_hold + dat.video_offset


                    # print("adjusted offset: " + str(dat.video_offset))
        ##                print(dat.video_offset)
                    # print(vid_start_date)
                    # print(dat.vid_start_date)

                    # Find vid start date in IMU data
                    files = sorted(os.listdir(dat.wd))
                    video_name = files[dat.v_num]
                    print("video name: " + video_name)


                    vn =  os.path.splitext(video_name)[0]

                    vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)

                    dat.vt["vid_start_date"][vid_match] = dat.vid_start_date
                    print(dat.vt["vid_start_date"][vid_match])
                    dat.vt[dat.time_col_string][vid_match] = dat.vid_start_date - dat.video_offset
                    print(dat.vid_start_date - dat.video_offset)
                    dat.vt["video_offset"][vid_match] = dat.video_offset.total_seconds()
                    print(dat.video_offset.total_seconds())
                    dat.vt["set"][vid_match] = 1

                    vt = dat.vt
                    if dat.view_only == False:
                        vt.to_csv(dat.wd + "/vt.csv",index = False)
                    else:
                        print("view only mode file not saved")


                    # dur = dat.frame_count/dat.fps
                    idx_start2 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
                    # print(dat.df.iloc[idx_start2,dat.time_col])
                    idx_end2 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick
                    # print(dat.df.iloc[idx_end2,dat.time_col])
                    dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid")] = video_name#dat.v_num

                    IMU_range = idx_end2 - idx_start2
                    vid_match = np.linspace(0,dat.frame_count,IMU_range)
                    dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("frame")] = vid_match

                    dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid_time")] = dat.df.iloc[np.arange(idx_start2,idx_end2),dat.time_col].dt.floor('s') - dat.video_offset

                    dat.df.iloc[np.arange(idx_start2,idx_start2 + (pce_hold.shape[0])),dat.df.columns.get_loc(dat.current_label_col)] = pce_hold
                    #Set colour of SYNC button
                    dat.btn_SYNC.configure(bg = 'green')
                    # pass

                w2.destroy()

            btn_sync = tk.Button(w2,text = "Video SYNC",command = lambda:fsynch(self,dat))
            btn_sync['state'] = "disabled"
            btn_sync.pack()

            def d1a(self,dat):
                btn_d1a['state'] = "disabled"
                btn_d1b['state'] = "normal"
                dat.d1a = dat.frame
            btn_d1a = tk.Button(w2,text = "Dive 1 START",command = lambda:d1a(self,dat))
            btn_d1a['state'] = "normal"
            btn_d1a.pack()

            def d1b(self,dat):
                btn_d1b['state'] = "disabled"
                btn_d2a['state'] = "normal"
                dat.d1b = dat.frame
            btn_d1b = tk.Button(w2,text = "Dive 1 END",command = lambda:d1b(self,dat))
            btn_d1b['state'] = "disabled"
            btn_d1b.pack()

            def d2a(self,dat):
                btn_d2a['state'] = "disabled"
                btn_d2b['state'] = "normal"
                dat.d2a = dat.frame
            btn_d2a = tk.Button(w2,text = "Dive 2 START",command = lambda:d2a(self,dat))
            btn_d2a['state'] = "disabled"
            btn_d2a.pack()

            def d2b(self,dat):
                btn_d2b['state'] = "disabled"
                btn_d3a['state'] = "normal"
                dat.d2b = dat.frame
            btn_d2b = tk.Button(w2,text = "Dive 2 END",command = lambda:d2b(self,dat))
            btn_d2b['state'] = "disabled"
            btn_d2b.pack()

            def d3a(self,dat):
                btn_d3a['state'] = "disabled"
                btn_d3b['state'] = "normal"
                dat.d3a = dat.frame
            btn_d3a = tk.Button(w2,text = "Dive 3 START",command = lambda:d3a(self,dat))
            btn_d3a['state'] = "disabled"
            btn_d3a.pack()

            def d3b(self,dat):
                btn_d3b['state'] = "disabled"
                btn_sync['state'] = "normal"
                dat.d3b = dat.frame
            btn_d3b = tk.Button(w2,text = "Dive 3 END",command = lambda:d3b(self,dat))
            btn_d3b['state'] = "disabled"
            btn_d3b.pack()

            def dreset(self,dat):
                btn_d1a['state'] = "normal"
                btn_d1b['state'] = "disabled"
                btn_d2a['state'] = "disabled"
                btn_d2b['state'] = "disabled"
                btn_d3a['state'] = "disabled"
                btn_d3b['state'] = "disabled"
                btn_sync['state'] = "disabled"
            btn_dreset = tk.Button(w2,text = "RESET",command = lambda:dreset(self,dat))
            btn_dreset.pack()

            def dclose(self,dat):
                w2.destroy()
            btn_dclose = tk.Button(w2,text = "CANCEL",command = lambda:dclose(self,dat))
            btn_dclose.pack()

    def synch_dives_auto(self,dat):
        pre,ext = os.path.splitext(dat.vid1)
        squash_file = pre + "_squash.jpg"
        if os.path.exists(squash_file):
            img= cv2.imread(squash_file,1)
            if img.shape[1] < img.shape[0]:
                showinfo(title='Image ERROR',message="ERROR - image shape seems to be incorrect")
                print("ERROR - image shape seems to be incorrect")
            else:
                #Split the colour image into channels (blue, green, red)
                b, g, r = cv2.split(img)
                #Weonly use the red channel and blur it to reduce noise
                blur = cv2.GaussianBlur(r,(11,11),0)  #blur image to reduce noise
                #Save the frame rate
                fps = dat.fps
                #Choose a window to smooth the data over
                window = 10
                #Specificy the minimum dive duration (i.e. minimum time between dive starts)
                dive_duration = 20

                #Get a rollin mean of the red channel values over the chosen window
                for row in range(blur.shape[0]):
                    image_mean = pd.DataFrame(blur[row,:]).rolling(fps*window).mean()

                #Create a dataframe of the values and assign frame numbers
                ts_diff = pd.DataFrame(image_mean.values,columns = ["val"]) #Smoothed values of the red channel
                #Get the slope of the values
                ts_diff['val_slope'] = ts_diff['val'].diff()
                ts_diff['raw'] = blur[0,:]                                 #Raw values
                ts_diff['frame'] = range(len(ts_diff))                      #Frame number
                ts_diff['dive'] = 0                                         #Dive start indicator (0 or 1)
                ts_diff['seconds'] = pd.to_timedelta(ts_diff.frame/fps, unit = 's')
                #Calculate the timestamp by using the creation date of the file
                ts_diff[dat.time_col_string] = pd.to_datetime(dat.vid_last_date ,unit = 's',utc = utc) + ts_diff['seconds']

                #Normalize the values for later plotting
                ts_diff['val_norm'] = (ts_diff['val'] - np.min(ts_diff['val'])) / (np.max(ts_diff['val']) - np.min(ts_diff['val']))


                #Find values where the smoothed values go below a threshold
                small_th = 50
                small_val = np.where(ts_diff.val < small_th)[0]

                #We also set an upper threshold where values have to cross before a new low value can be chosen
                large_th = 100

                #Assign identifiers to monitor if values are within our chosen margins (see for-loop below)
                prev_dive = 0
                next_high = 0

                #An empty list to keep the row indices where we identify dive starts
                dive_starts = []
                ts_diff['dive'] = 0
                # dive_th = []
                # dive_i = []

                #Loop through all the small values and see if they are elligible to be dive starts
                for i in range(len(small_val)-1):
                    row = small_val[i]
                    #Continue if the value has increased sufficiently
                    if row > next_high:
                        dur = row - prev_dive
                        #We only save the row index if the value changed from high (large_th) to low (small_th) and the duration to the previous valid point is long enough (dive_dur)
                        if (prev_dive == 0) or (dur > fps*dive_duration):
                             #Now we look for the poit where the dive starts
                             #Either the maximum pixel value prior to the detected change point
                            # i_start = np.where(ts_diff.val[:row] == ts_diff.val[:row].max() )
                            # i_start = ts_diff['val'][prev_dive:row].idxmax()
                            # dive_starts.append(i_start)
                            # dive_th.append(row)
                            # dive_i.append(i)
                            # prev_dive = row
                            # ts_diff.dive[i_start] = 1
                            #OR
                            #We look for a change in slope - i.e. rapid change in light conditions as the bird dives
                            for islope in range(row,prev_dive,-1):
                                # print(islope)
                                # print(ts_diff.val_slope[islope])
                                if ts_diff.val_slope[islope] > -0.25:
                                    # print(islope)
                                    i_start = islope

                                    dive_starts.append(i_start)
                                    # dive_th.append(row)
                                    # dive_i.append(i)
                                    prev_dive = row
                                    ts_diff.loc[i_start,"dive"] = 1
                                    break


                            try:
                                next_high = ts_diff.val[row:len(ts_diff)] > large_th
                                next_high = next_high[next_high == True].index[0]
                            except:
                                break
                #Finally - we get the time difference between the identified dive starts to compare to the TDR data
                vdiff = ts_diff[ts_diff.dive == 1]
                #Save the time of the seconds dive start (the first dive start will have a diff value of 0 and will be skipped below)
                v_frame = vdiff.reset_index().loc[1,'frame']
                #Now calculate the time difference between the dive starts
                vdiff = vdiff[dat.time_col_string].diff().dt.total_seconds().fillna(0).astype(int)
                # vdiff = np.round(np.diff(ts_diff[ts_diff['dive'] == 1].index/fps))

                print("Video dive times")
                print(vdiff)

                #Get dive start times from TDR data
                # dive_df = dat.df.iloc[dat.vid_idx_start:dat.vid_idx_end,:]
                # print("TDR dive times")
                # # print(np.diff(dive_df[(dive_df['dpoint'] == 1) and (dive_df['dive_max_depth'] > 3) and (dive_df['dive_dur'] > 20)].index/25))

                dive_df = dat.df[dat.df['dpoint'] == 1 ].reset_index()
                # dive_df = dive_df[dive_df['dive_max_depth'] > 3].reset_index()
                dive_df = dive_df[dive_df['dive_dur'] > 20].reset_index()

                tdrdiff = dive_df[dat.time_col_string].diff().dt.total_seconds().fillna(0).astype(int)
                # tdrdiff = np.round(np.diff(dive_df.index/dat.sampling_rate))

                min_diff = 9999
                i_min = 0
                for i in range(len(tdrdiff) - (len(vdiff)-1)):
                    i_diff = (np.mean(abs(tdrdiff[i:i+(len(vdiff)-1)].values - vdiff[1:len(vdiff)].values)))
                    if i_diff < min_diff:
                        min_diff = i_diff
                        i_min = i
                # print(min_diff)
                # print(tdrdiff[i_min:i_min+(len(vdiff)-1)].values)
                df_row = dat.df[dat.df[dat.time_col_string] == dive_df.loc[i_min, dat.time_col_string]].index.tolist()[0]
                # mdiff = np.abs(np.round(np.mean(vdiff[range(min(len(tdrdiff),len(vdiff)))]-tdrdiff[range(min(len(tdrdiff),len(vdiff)))])))
                if min_diff <= 1:

                    #point the IMU to the right point
                    dat.IMU_date = dat.df.loc[df_row,dat.time_col_string]
                    print(f"IMU Date: {dat.IMU_date}")

                    #point the vid to the right point
                    # dat.vid.set(1,v_frame)
                    dat.frame = v_frame
                    print(f"Video Frame: {dat.frame}")
                    Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()
                    print(f"Video Start Date: {dat.vid_start_date}")
                    print(f"Video Frame Date: {dat.frame_date}")

                    #Set the time
                    Button_functions.set_time(dat)
        #             #Synchronise the time
        #             dat.date_set = 1
        #             print('synch video and IMU')
        #             # print(dat.IMU_date)
        #             # print(dat.vid_start_date)
        #
        #             dur = dat.frame_count/dat.fps
        #
        #             idx_start1 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
        #             idx_end1 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick
        #
        #             pce_hold = dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc("PCE")]
        #             dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc("PCE")] = 0
        #
        #             offset_hold = dat.video_offset
        #             print("holding offset: " + str(offset_hold))
        #             dat.video_offset = dat.IMU_date - dat.frame_date
        #             print("new offset: " + str(dat.video_offset))
        #             vid_start_date = dat.vid_start_date + dat.video_offset
        #             dat.vid_start_date = vid_start_date
        #             dat.video_offset = offset_hold + dat.video_offset
        #
        #
        #             # print("adjusted offset: " + str(dat.video_offset))
        # ##                print(dat.video_offset)
        #             # print(vid_start_date)
        #             # print(dat.vid_start_date)
        #
        #             # Find vid start date in IMU data
        #             files = sorted(os.listdir(dat.wd))
        #             video_name = files[dat.v_num]
        #             print("video name: " + video_name)
        #
        #             # dat.vt.iloc[dat.vt["vid"].get_loc(dat.video_name),2] = dat.vid_start_date
        #             vn =  os.path.splitext(video_name)[0] + ".mp4"
        #
        #             dat.vt.vid_start_date[dat.vt["vid"] == vn] = dat.vid_start_date
        #             print(dat.vid_start_date)
        #             dat.vt.Timestamp[dat.vt["vid"] == vn] = dat.vid_start_date - dat.video_offset
        #             print(dat.vid_start_date - dat.video_offset)
        #             dat.vt.video_offset[dat.vt["vid"] == vn] = dat.video_offset.total_seconds()
        #             print(dat.video_offset.total_seconds())
        #             dat.vt.set[dat.vt["vid"] == vn] = 1
        #
        #             # vt = dat.vt
        #             # vt.to_csv(dat.wd + "/vt.csv",index = False)
        #
        #             # dur = dat.frame_count/dat.fps
        #             idx_start2 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
        #             print(dat.df.iloc[idx_start2,dat.time_col])
        #             idx_end2 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick
        #             print(dat.df.iloc[idx_end2,dat.time_col])
        #             dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid")] = video_name#dat.v_num
        #
        #             IMU_range = idx_end2 - idx_start2
        #             vid_match = np.linspace(0,dat.frame_count,IMU_range)
        #             dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("frame")] = vid_match
        #
        #             dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid_time")] = dat.df.iloc[np.arange(idx_start2,idx_end2),dat.time_col].dt.floor('s') - dat.video_offset
        #
        #             dat.df.iloc[np.arange(idx_start2,idx_start2 + (pce_hold.shape[0])),dat.df.columns.get_loc("PCE")] = pce_hold

                else:
                    print("could not find match")


        else:
            showinfo(title='Synch ERROR',message="Squash the video file first")
    def pce_navigate(self,dat):
        global c,window#,bpre,bpost #this counts the number of annotations avaiable

        #Create a new window with options
        wa1 = tk.Toplevel(self)
        wa1.title("Image export")
        wa1.geometry("800x1000")
        wa1.attributes('-topmost',True)
        def fclose(self,dat):
            wa1.destroy()
        #Close button
        btn_close = tk.Button(wa1,text = "Close",command = lambda:fclose(self,dat))
        btn_close.pack()


        #Function to move to next PCE
        def next_pce(self,dat,sselect):
            global c,window#,bpre,bpost

                #Check to see how many annotations were chosen
            temp_ann = [] #Holder for annotation choices

            #Create folder

            #Read the tick boxes (created from the available annotations)
            for i in range(c):
                # print(i)
                x = eval("dat.ann" + str(i))
                #Continue if a tickbox is chosen
                if(x.get() > 0):
                    print(x.get())
                    temp_ann.append(x.get())

            #Find where we are in the data
            idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick

            #Get the selected PCE data from the current point onwards
            if sselect == 'next':
                temp_dat = dat.df.loc[np.arange((idx+2),len(dat.df)-1)]
                # temp_dat = dat.df.loc[np.arange((idx+2),dat.vid_idx_end)]
            elif sselect == 'prev':
                temp_dat = dat.df.loc[np.arange(0,(idx-2))]
                # temp_dat = dat.df.loc[np.arange(dat.vid_idx_start,(idx-2))]

            temp_dat = temp_dat[temp_dat[dat.current_label_col].isin(temp_ann)].reset_index()
            print(str(len(temp_dat)) + " annotation present")

            #Choose the next available annotations and check if the video file is correct
            if sselect == 'next':
                pce_select = temp_dat.iloc[0]
                #Check video
                if pce_select.vid != (dat.vid1.split('/')[-1].split(".")[0]+'.mp4'):
                    # pass
                    dat.view_only = True
                    dat.vid1 = dat.wd + "/"+ pce_select.vid
                    Menu_functions_VIDEO.load_avi(self,dat)
                    dat.view_only = False

                print("pce_vid: " +dat.wd  +pce_select.vid)
            elif sselect == 'prev':
                pce_select = temp_dat.iloc[-1]
                if pce_select.vid != (dat.vid1.split('/')[-1].split(".")[0]+'.mp4'):
                    # pass
                    dat.view_only = True
                    dat.vid1 = dat.wd + "/"+ pce_select.vid
                    Menu_functions_VIDEO.load_avi(self,dat)
                    dat.view_only = False


                print("pce_vid: " +dat.wd  +pce_select.vid)
            #Jump to the frame
            dat.frame = dat.frame + (pce_select[dat.time_col_string] - dat.frame_date).total_seconds()*dat.fps
            #Find the row for the selected annotation
            idx = (np.abs(dat.df.iloc[:,dat.time_col] - pce_select[dat.time_col_string])).argmin() #Find the row in the df with the nearest value to ipick

            #Update the plots
            dat.vid.set(1,dat.frame)
            Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()

            ipick = dat.df.loc[idx,dat.time_col_string]

            dat.sub_min = idx -50
            dat.sub_max = idx + 50
            dat.zoom_ipick = ipick
##            frame_num = int(frame_count*id_perc)
##            image = plotframe(frame_num)
##            panel.configure(image = image)
##            panel.image = image
            dat.vline3.remove()
            dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')

            dat.ax_zoom.cla() #Clear axes
            # Extracting the data for zoom plot
            #Specificy the rows to plot
            row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
            # Plot each column individually
            for col in dat.p_cols:
                # Select the data for the current column
                x_data = dat.df.iloc[row_to_use, dat.time_col]
                y_data = dat.df.iloc[row_to_use, col]

                # Exclude NA values
                non_na_mask = ~y_data.isna()
                x_data = x_data[non_na_mask]
                y_data = y_data[non_na_mask]

                # Plot the current column
                dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
            # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
            dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
            dat.figure_zoom.canvas.draw() #Redraw the figure

            #Update Audio plot
            if dat.audio_present == True:
                dat.sub_window = dat.sub_max - dat.sub_min
                dat.aud_window = int((dat.sub_window*640)/2)
                dat.aud_point = int(dat.frame*640)
                if dat.frame > 1:
                    dat.aud_min = dat.aud_point - dat.aud_window
                else:
                    dat.aud_min = dat.aud_point
                dat.aud_max = dat.aud_point + dat.aud_window

                dat.ax_audio.cla()
                dat.ax_audio.plot(dat.audio.iloc[np.arange(dat.aud_min,dat.aud_max,10),1],dat.audio.iloc[np.arange(dat.aud_min,dat.aud_max,10),0])
                dat.vline_aud = dat.ax_audio.axvline(x = dat.audio.iloc[dat.aud_point,1],color = 'red')
                dat.figure_audio.canvas.draw()

            self.update()
            #end next_pce


        #NExt PCE button
        btn_next_pce = tk.Button(wa1,text = "Next PCE", command = lambda:next_pce(self,dat,sselect = 'next')).pack()
        btn_next_pce = tk.Button(wa1,text = "Previous PCE", command = lambda:next_pce(self,dat,sselect = 'prev')).pack()

        #Set initial location for tickboxes (listing all available annotations)
        c = 0
        ix = 100
        iy = 50

        tk.Label(wa1, text="Select annotations").place(x = ix, y=0)
        #Loop over all annotations and create a tickbox as well as a variable storing the choices
        for i in np.unique(pd.to_numeric(dat.df[dat.current_label_col])):
            if i > 0: #We skip 0 annotation
                #Create the varialble where the tickbox value is stored
                exec("dat.ann" + str(c) +" = tk.IntVar()")
                # exec("dat.annf" + str(c) +" = tk.IntVar()")
                #Create the tickbox
                tk.Checkbutton(wa1, text=i,variable=eval("dat.ann" + str(c)), onvalue=i, offvalue=0 ).place(x = ix, y=iy)
                #Increment the counter of annotations and coordinates of the next box
                c = c +1
                iy = iy + 50
                #If we reach the end of the window, skip to the next column (should not be neccesary for annotations)
                if iy > 1000:
                    iy = 50
                    ix = ix + 150
        #end pce_navigate
    def check_horison_column(self,dat):
        if "horison_angle" not in dat.df.columns:
            dat.df["horison_angle"] = None

class Menu_functions_EXPORT(Data_Frame):
    def export_events(self,dat):
        pre,ext = os.path.splitext(dat.filename)
        out_file = pre + "_OUT.csv"
        dat.df[dat.current_label_col] = pd.to_numeric(dat.df[dat.current_label_col],errors='coerce')
        pce_out = dat.df[[dat.current_label_col,"vid","frame","vid_time","BEHAV"]]

        #Create a new window to show a prompt to wait
        #New temp window
        wtemp = tk.Toplevel(self)
        wtemp.title("Loading data")
        wtemp.geometry("400x200")
        wtemp.attributes('-topmost',True)

        label = tk.Label(wtemp, text="Please wait - this may take a while")  # create the label
        label.pack()  # add the label to the window
        self.update()
        # dat.df.to_csv(out_file ,index = False)
        pce_out.to_csv(out_file,index = False)
        wtemp.destroy()
        with open('config.txt','w') as f:
            f.write(str(dat.vid_start_date)+
                    str(dat.filename)+
                    str(dat.vid1)+
                    str(dat.frame)
                    )
            f.write('\n')
        dat.bexported = True #Set flag
##            f.close()
        print("DONE")
        # wtemp.destroy()
        # self.update()
    def export_config(self):
        pre,ext = os.path.splitext(self.filename)
##        out_file = pre + "_CONFIG.txt"
##
##        with open(out_file,'w') as f:
##            f.write(str(self.vid_start_date)+ ","+
##                    str(self.filename)+ ","+
##                    str(self.vid1)+ ","+
##                    str(self.frame)
##                    )
##            f.write('\n')
####            f.close()
##        print("DONE")


        temp = temp_Data_Frame()
        temp.filename = self.filename
        temp.dive_num = self.dive_num
        temp.col_names = self.col_names
        temp.vid1 = self.vid1
        temp.vid_last_date = self.vid_last_date
        temp.frame = self.frame
        temp.frame_count = self.frame_count
        temp.fps = self.fps
        temp.vid_start_date = self.vid_start_date
        temp.frame_date = self.frame_date
        temp.date_set = self.date_set
        temp.video_loaded = self.video_loaded
        temp.wd = self.wd
        temp.v_num = self.v_num
        temp.video_offset =  self.video_offset
        temp.vt_present = self.vt_present
        temp.vt = self.vt

        # vt = temp.vt
        # vt.to_csv(self.wd + "/vt.csv",index = False)

        out_file = pre + "_CONFIG.pkl"
        with open(out_file, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(temp, outp, pickle.HIGHEST_PROTOCOL)
        print("DONE")
        self.bconfig = True
    def export_annotated(self,dat):
        global c,window#,bpre,bpost #this counts the number of annotations avaiable

        #Create a new window with options
        wa1 = tk.Toplevel(self)
        wa1.title("Image export")
        wa1.geometry("800x1000")
        wa1.attributes('-topmost',True)
        def fclose(self,dat):
            wa1.destroy()
        #Close button
        btn_close = tk.Button(wa1,text = "Close",command = lambda:fclose(self,dat))
        btn_close.pack()

        #Export ALL button
        #Function to export annotations from all videos
        def fall(self,dat,sselect):
            global c,window#,bpre,bpost

                #Check to see how many annotations were chosen
            temp_ann = [] #Holder for annotation choices

            #Create folder
            if not os.path.exists(dat.wd + "/PCE_images/"):
                os.mkdir(dat.wd +  "/PCE_images/")

            #Read the tick boxes (created from the available annotations)
            for i in range(c):
                # print(i)
                x = eval("dat.ann" + str(i))
                #Continue if a tickbox is chosen
                if(x.get() > 0):
                    # print(x.get())
                    temp_ann.append(x.get())
                    #Create folder for the images
                    if not os.path.exists(dat.wd + "/PCE_images/" + str(x.get())):
                        os.mkdir(dat.wd + "/PCE_images/" + str(x.get()))
            # print("exporting:")
            # print(temp_ann)

            print("Exporting images")
            #Check to see if a window has been chosen
            if window.get() > 1:
                print('window: ' + str(window.get()))
                if sselect == 'all':
                    print("all")
                    trows = dat.df[dat.df[dat.current_label_col].isin(temp_ann)].index
                    temp_df = dat.df.copy() #holder of the df to expand the annotations to the desired window
                elif sselect == 'single':
                    print("single")
                    trows = dat.df[dat.df.loc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.current_label_col].isin(temp_ann)].index
                    temp_df = dat.df[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1)].copy() #holder of the df to expand the annotations to the desired window

                # print(len(temp_df))
                trows2 = [] #holder for the rows selected
                # print("bpre" + str(dat.bpre.get()))
                # print("bpost" + str(dat.bpost.get()))
                for row in trows:
                    #Export images before and after annotation
                    if (dat.bpre.get() == 1) and  (dat.bpost.get() == 1):
                        for tval in range(row-window.get(),row+window.get()+1,1):
                            trows2.append(tval)
                            temp_df.loc[(row-window.get()):(row+window.get()+1),dat.current_label_col] = temp_df.loc[row,dat.current_label_col]
                    #Export images before annotation
                    elif dat.bpre.get() == 1:
                        for tval in range(row-window.get(),row+1,1):
                            trows2.append(tval)
                            temp_df.loc[(row-window.get()):(row+1),dat.current_label_col] = temp_df.loc[row,dat.current_label_col]
                    #Export images after annotation
                    elif dat.bpost.get() ==1:
                        for tval in range(row,row+window.get()+1,1):
                            trows2.append(tval)
                            temp_df.loc[row:(row+window.get()+1),dat.current_label_col] = temp_df.loc[row,dat.current_label_col]
                    #Revert to single image if before/after not specified
                    else:
                        print("Please select -before- and/or -after- if you want a window")
                        tval = row
                        trows2.append(tval)

                # print(trows2)
                temp_dat = temp_df.iloc[trows2,:].reset_index()
            else:
                temp_dat = dat.df[dat.df[dat.current_label_col].isin(temp_ann)].reset_index()

            #Print the number of annotations found
            print(str(len(temp_dat)) + " annotations found")
            #Prompt user to make sure
            res = tk.messagebox.askyesno("Export images from ALL videos",str(len(temp_dat)) + " annotations found. "+"This may take some time. Continue?")
            if res:

                files = sorted(os.listdir(dat.wd))
                for row in range(len(temp_dat)):
                    # row = 0

                    if not temp_dat.loc[row,"vid"] is np.nan:
                        vid = (temp_dat.loc[row,"vid"]).split(".")[0] + ".mp4"
                        frame = int(temp_dat.loc[row,"frame"])
                        vid_full = list(filter(lambda x: vid in x, files))[0]
                        cap = cv2.VideoCapture(dat.wd +  "/"+ vid_full)
                        cap.set(1,frame)
                        s, image = cap.read()
                        filename = dat.wd +  "/PCE_images/" + str(temp_dat.loc[row,dat.current_label_col]) + "/" + str(temp_dat.loc[row,"vid"]) + "-" + str(frame) + ".jpg"
                        if s:
                            cv2.imwrite(filename, image)
                            print(filename)
                        else:
                            print("can't write" + filename)
                print("Done Exporting")


        btn_export_all = tk.Button(wa1,text = "Export ALL", command = lambda:fall(self,dat,sselect = 'all')).pack()

        #Export ALL button
        btn_export_single = tk.Button(wa1,text = "Export current", command = lambda:fall(self,dat,sselect = 'single')).pack()

        #Set initial location for tickboxes (listing all available annotations)
        c = 0
        ix = 100
        iy = 50

        tk.Label(wa1, text="Select annotations").place(x = ix, y=0)
        #Loop over all annotations and create a tickbox as well as a variable storing the choices
        for i in np.unique(pd.to_numeric(dat.df[dat.current_label_col])):
            if i > 0: #We skip 0 annotation
                #Create the varialble where the tickbox value is stored
                exec("dat.ann" + str(c) +" = tk.IntVar()")
                # exec("dat.annf" + str(c) +" = tk.IntVar()")
                #Create the tickbox
                tk.Checkbutton(wa1, text=i,variable=eval("dat.ann" + str(c)), onvalue=i, offvalue=0 ).place(x = ix, y=iy)
                #Increment the counter of annotations and coordinates of the next box
                c = c +1
                iy = iy + 50
                #If we reach the end of the window, skip to the next column (should not be neccesary for annotations)
                if iy > 1000:
                    iy = 50
                    ix = ix + 150

        #Create slider to select window (OPTIONAL choice)
        iy = iy + 50
        tk.Label(wa1, text="Select window").place(x = ix, y=iy)
        iy = iy + 25
        window = tk.Scale(wa1, from_=1, to=25, orient="horizontal")
        window.place(x = ix, y=iy)

        #Create buttons to select wether to export the window, prior to and/or after
        dat.bpre = tk.IntVar()
        iy = iy + 50
        cpre = tk.Checkbutton(wa1, text="Before event",variable=dat.bpre , onvalue=1, offvalue=0)
        cpre.place(x = ix, y=iy)

        # tk.Checkbutton(wa1, text="Before event",variable=dat.bpre, onvalue=1, offvalue=0 ).place(x = ix, y=iy)

        dat.bpost = tk.IntVar()
        iy = iy + 50
        cpost = tk.Checkbutton(wa1, text="After event",variable=dat.bpost, onvalue=1, offvalue=0)
        cpost.place(x = ix, y=iy)

        # tk.Checkbutton(wa1, text="After event",variable=dat.bpost, onvalue=1, offvalue=0 ).place(x = ix, y=iy)
    def export_dive_images(self,dat):
        global c,window#,bpre,bpost #this counts the number of annotations avaiable

        #Create a new window with options
        wa1 = tk.Toplevel(self)
        wa1.title("Image export")
        wa1.geometry("800x1000")
        wa1.attributes('-topmost',True)
        def fclose(self,dat):
            wa1.destroy()
        #Close button
        btn_close = tk.Button(wa1,text = "Close",command = lambda:fclose(self,dat))
        btn_close.pack()

        #Export ALL button
        #Function to export annotations from all videos
        def fall(self,dat):
            temp_dat = dat.df[dat.df['forage_dive']==1].reset_index()

            #Print the number of annotations found
            print(str(len(temp_dat)) + " annotations found")
            #Prompt user to make sure
            res = tk.messagebox.askyesno("Export images from ALL videos",str(len(temp_dat)) + " annotations found. "+"This may take some time. Continue?")
            if res:
                dat.bexport = True #flag to show we are busy Exporting

                #Create a new window to show a progress bar and cancel button
                #New temp window
                w_prog = tk.Toplevel(self)
                w_prog.title("Exporting images")
                w_prog.geometry("400x200")
                label = tk.Label(w_prog, text="Please wait - this may take a while")  # create the label
                w_prog.wm_transient(self)
                label.pack()  # add the label to the window
                def fcancel(self,dat):
                    dat.bexport = False
                    w_prog.destroy()
                #Place a cancel button
                btn_close = tk.Button(w_prog,text = "Cancel",command = lambda:fcancel(self,dat))
                btn_close.pack()

                #Initiate the progress bar
                pb = ttk.Progressbar(
                 w_prog,
                 orient='horizontal',
                 mode='determinate',
                 length=100
                 # value = 0
                 )
                pb.pack()
                # #Loop through all the frames and save the first column (pixels) to the empty image
                print("Finding dives...")

                #Create the folder if not already present
                if not os.path.exists(dat.wd + "/Annotated_images/"):
                    os.mkdir(dat.wd +  "/Annotated_images/")

                files = sorted(os.listdir(dat.wd))
                icount = 0
                a_rows = [] #holder for rows
                for row in range(len(temp_dat)):
                    if dat.bexport == True:
                        icount = icount+1
                        exp_proportion = round(icount / len(temp_dat),2)*100
                        pb['value'] = exp_proportion
                        w_prog.update_idletasks()
                        self.update()
                        # row = 0

                        # if not temp_dat.loc[row,"vid"] is np.nan:
                        if not pd.isnull(temp_dat.loc[row,"vid"]):
                            a_rows.append(row)
                            vid = str(temp_dat.loc[row,"vid"]).split(".")[0] + ".mp4"
                            frame = int(temp_dat.loc[row,"frame"])
                            vid_full = list(filter(lambda x: vid in x, files))[0]
                            cap = cv2.VideoCapture(dat.wd +  "/"+ vid_full)
                            cap.set(1,frame)
                            s, image = cap.read()
                            filename = dat.wd +  "/Annotated_images/"  + str(temp_dat.loc[row,"vid"]) + "-" + str(frame) + "_" + str(temp_dat.loc[row,'PCE_exp']) + ".jpg"
                            if s:
                                cv2.imwrite(filename, image)
                                # print(filename)
                            else:
                                print("can't write" + filename)

                temp_dat.loc[a_rows,['vid','frame','PCE_exp']].to_csv(dat.wd +  "/Annotated_images/labels.csv",index = False)
                print("Done Exporting")
                dat.bexport = False
                w_prog.destroy()

        btn_export_all = tk.Button(wa1,text = "Export images", command = lambda:fall(self,dat)).pack()
    # tk.Checkbutton(wa1, text="After event",variable=dat.bpost, onvalue=1, offvalue=0 ).place(x = ix, y=iy)

    def export_yolo(self,dat):
        #Loop through all the videos and export where we find images with predictions
        # #Loop through all the frames and predict while saving the output to the dataframe
        print("YOLO predicting in progress...")
        dat.byolo = True #Flag to say we are busy with the yolo model
        # For running all files
        v_num = -1       #video location in the folder
        files = sorted(os.listdir(dat.wd)) #All files within the video directory
        # print(files)
        #Loop through all the video files
        while (v_num < len(files)) and dat.byolo == True:
                v_num = v_num + 1 #Increment the file location
                #Look for the next mp4 file
                while(((not files[v_num].lower().endswith(".mp4")))  and v_num < (len(files)-1)):
                    v_num = v_num + 1
                #Stop if no more mp4 files are present
                if not files[v_num].lower().endswith(".mp4"):
                    break
                filename = dat.wd +'/' + files[v_num]
                print(filename)
                #Squash image
                if dat.byolo == True:
                    dat.vid1 = filename
                    dat.vid = cv2.VideoCapture(dat.vid1)
                    dat.fps = int(round(dat.vid.get(cv2.CAP_PROP_FPS)))
                    dat.frame_count = int(dat.vid.get(cv2.CAP_PROP_FRAME_COUNT))

                    #Get the timestamp time from the video metadata
                    #First - Run ffprobe command to get video metadata
                    ffprobe_command = [
                        'ffprobe',
                        '-v', 'quiet',
                        '-print_format', 'json',
                        '-show_format',
                        dat.vid1
                    ]

                    try:
                        ffprobe_output = subprocess.check_output(ffprobe_command, stderr=subprocess.STDOUT)
                        metadata = json.loads(ffprobe_output.decode('utf-8'))
                        if 'creation_time' in metadata['format']['tags']:
                            creation_time = metadata['format']['tags']['creation_time']
                            dat.vid_last_date = pd.to_datetime(creation_time,utc = utc).timestamp()# - timedelta(seconds = dat.frame_count/dat.fps)
                            # print("last date (i.e. creation time): " + str(pd.to_datetime(creation_time,utc = utc)))
                        else:
                            print('no creation time')
                            #Otherwise check the creation date using os.path
                            dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
                    except subprocess.CalledProcessError as e:
                        print("Error:", e)
                        dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX


                    files = sorted(os.listdir(dat.wd))
                    vn = dat.vid1.split("/")[-1]
                    dat.v_num = files.index(vn)
                    pre,ext = os.path.splitext(filename)
                    dat.frame = 0
                    video_name = files[dat.v_num]
                    vn =  os.path.splitext(video_name)[0]
                    # dat.v_num = dat.v_num+1
                    # while(((not files[dat.v_num].lower().endswith(".avi")) and (not files[dat.v_num].lower().endswith(".mov")))  and dat.v_num < len(files)):
                    #     dat.v_num = dat.v_num + 1
                    #
                    # print("##############DEBUG##################")
                    # print(int(dat.vt["set"][vid_match]))
                    # print(vn)
                    # print(dat.vt['vid'])

                    vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)
                    if(int(dat.vt["set"][vid_match]) == 1): #if(int(dat.vt["set"][vid_match]) == 1):

                        dat.vid_date_set = 1
                        dat.vid_start_date = dat.vt["vid_start_date"][vid_match]#.values
                        # print(dat.vt["vid_start_date"][vid_match])
                        # print(f"Video start date: {dat.vid_start_date}")
                        dat.video_offset = dat.vid_start_date - dat.vt["Timestamp"][vid_match]#.values #timedelta(seconds = int(dat.vt["video_offset"][vid_match]))
                        # print("offset: " + str(dat.video_offset))
                        # print("File offset: " + str(dat.vt["video_offset"][vid_match]))
                        # print(dat.video_offset)
                    else:
                        dat.vid_date_set = 0
                        # dat.vid_start_date = pd.to_datetime(dat.vid_last_date -(dat.frame_count/dat.fps),unit = 's',utc = utc)
                        dat.vid_start_date = pd.to_datetime(dat.vid_last_date ,unit = 's',utc = utc)
                        # print("original date: " + str(dat.vid_start_date))
                        vid_start_date = dat.vid_start_date + dat.video_offset
                        # print("offset: " + str(dat.video_offset))

                        dat.vid_start_date = vid_start_date
                        # print("adjusted date: " + str(dat.vid_start_date))

                    dat.frame_date=dat.vid_start_date

                    #First check if there are any dives
                    dat.vid_idx_start = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin()
                    dat.vid_idx_end = dat.vid_idx_start + int((dat.frame_count/dat.fps)*dat.frequency)
                    self.update()
                    if 1 in np.unique(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")]):#"1" in dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")].values:
                        indices = dat.df.index[df['dive'] == 1].tolist()
                        #Get the video name (without suffix)
                        pre,ext = os.path.splitext(files[v_num])
                        frame = 0
                        ##################################
                        #For running the loaded video only
                        ##################################

                        #Create a new window to show a progress bar and cancel button
                        #New temp window
                        w2 = tk.Toplevel(self)
                        w2.title("YOLO predicting")
                        w2.geometry("400x200")
                        label = tk.Label(w2, text="Please wait - this may take a while")  # create the label
                        label.pack()  # add the label to the window
                        def fcancel(self,dat):
                            print("YOLO model cancelled")
                            dat.byolo = False
                            w2.destroy()
                        #Place a cancel button
                        btn_close = tk.Button(w2,text = "Cancel",command = lambda:fcancel(self,dat))
                        btn_close.pack()

                        #Initiate the progress bar
                        pb = ttk.Progressbar(
                         w2,
                         orient='horizontal',
                         mode='determinate',
                         length=350
                         # value = 0
                         )
                        pb.pack()

                        dat.frame = 1
                        dat.vid.set(1,dat.frame)

                        while True:
                            #Update the progress bar
                             pb['value'] = (dat.frame/dat.frame_count)*100
                             w2.update_idletasks()
                             #Update the screen
                             self.update()

                             # Read a frame from the video
                             ret, image = dat.vid.read()
                             self.update()
                             #Get the frame ms time
                             time = float(dat.frame)/dat.fps
                             #Get the frame date and time
                             frame_date = dat.vid_start_date + timedelta(seconds = time)
                             #Get the closest point matching from the dataframe
                             idx = (np.abs(dat.df.iloc[:,dat.time_col] - frame_date)).argmin() #Find the row in the df with the nearest value to ipick
                             #Check if a dive is present
                             bdive = dat.df.iloc[idx,dat.df.columns.get_loc("dive")] #This will be a 0 or a 1

                             #Only continue if in a dive
                             if bdive:
                                 print(dat.frame)
                                 # # Read a frame from the video
                                 # ret, image = dat.vid.read()

                                 # print(frame)
                                 # pb['value'] = (frame/dat.frame_count)*100
                                 # print(frame_count)
                                 # print((frame/frame_count)*350)
                                 # w2.update_idletasks()


                                 # print(f'idx: {idx}')
                                 if ret:
                                     #Predict on the frame using the loaded model

                                     yolo_predict = dat.yolo_model.predict(image,
                                     conf = 0.3,
                                     iou = 0.6,
                                     nms = True,
                                     max_det = 50,
                                     seed = 42,
                                     verbose=False)


                                     #Read the results
                                     model_result = yolo_predict[0]


                                     #Find all the bounding boxes
                                     # bboxes = model_result.boxes.xyxy.int().tolist()
                                     # print(bboxes)
                                     bboxes2 = model_result.boxes.xywhn.tolist()
                                     #Extract the model results
                                     cls =  model_result.boxes.cls.int().tolist()
                                     #Extract the probabilities
                                     confs =  model_result.boxes.conf.tolist()
                                     #Mergee the boxes, labels and probs
                                     bb_cls_cf = [(bb,cl,cf) for cl,bb,cf in zip(cls,bboxes2,confs)]
                                     #Draw the boxes on the image
                                     # box_count = 0
                                     out_list = []
                                     for  bb,cl,cf in bb_cls_cf:
                                         b_x1,b_y1,b_x2,b_y2 =  bb
                                         out_row = [0,b_x1,b_y1,b_x2,b_y2]
                                         out_list.append(out_row)

                                         # cv2.rectangle(image,(b_x1,b_y1),(b_x2,b_y2), (0, 255, 0), 2)
                                         # cv2.putText(image, str(cf), (b_x1, b_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                         # box_count+=1
                                     if len(out_list) > 2:
                                         #Video file name
                                         fn_base = str(dat.vid1)
                                         #modified output filename
                                         fn_mod = os.path.dirname(fn_base) + "/" + fn_base.split("/")[-3].zfill(2) + "_" + os.path.basename(fn_base)

                                         fn = fn_mod + "_" + str(dat.frame)
                                         # print(fn)
                                         cv2.imwrite(fn+ "_0.jpg", image)
                                         # Writing the list to a tab-delimited file
                                         with open(fn+ "_0.txt", 'w') as file:
                                            for row in out_list:
                                                line = ' '.join(map(str, row))  # Convert each element to a string and join with tabs
                                                file.write(line + '\n')


                                     # #DEBUG
                                     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                     # image = cv2.resize(image,(dat.video_frame.winfo_width(),int(dat.video_frame.winfo_width()/1.5)))
                                     # image = Image.fromarray(image)
                                     # image = ImageTk.PhotoImage(image)
                                     # #wait to make sure the selected playback speed is adhered # TODO
                                     # dat.panel.configure(image = image)
                                     # dat.panel.image = image
                                     # self.update()

                             if (dat.frame >= dat.frame_count) or (dat.byolo == False):
                                 print("end of video")
                                 w2.destroy()
                                 break
                             else:
                                 dat.frame += 1


                        #For running all files
                        # v_num = -1       #video location in the folder
                        # files = sorted(os.listdir(dat.wd)) #All files within the video directory
                        # # print(files)
                        # #Loop through all the video files
                        # while (v_num < len(files)) and dat.byolo == True:
                        #         v_num = v_num + 1 #Increment the file location
                        #         #Look for the next mp4 file
                        #         while(((not files[v_num].lower().endswith(".mp4")))  and v_num < (len(files)-1)):
                        #             v_num = v_num + 1
                        #         #Stop if no more mp4 files are present
                        #         if not files[v_num].lower().endswith(".mp4"):
                        #             break
                        #         filename = dat.wd +'/' + files[v_num]
                        #         print(filename)


                        #Save the output
                        if dat.view_only == False:

                            print("YOLO SAVED")


                    else:
                        print("no dives present")
                        # print(np.unique(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")]))
                        # print(1 in np.unique(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")]))
                        # print("1"in np.unique(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")]))
                        # print(1 in dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")].values)
                        # print("1" in dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),1),dat.df.columns.get_loc("dive")].values)

class Menu_functions_ANNOTATE(Data_Frame):
    def choose_annotation_col(self,dat):
    #Create a new column for annotations or select a previously created column
        pass

    def selection_warning(self,dat):
        #Check which option is chosen from the menu and print the approriate warning
        if dat.annotate_selection.get():
            showinfo(title='Annotate method',message='Annotation will be made BETWEEN SELECTED points')
        elif ~dat.annotate_selection.get():
            showinfo(title='Annotate method',message='Annotation will be made on SINGLE points')

class Menu_functions_CHEATSHEETS(Data_Frame):
    def view_data(self,dat):

        foldername = fd.askdirectory(title = "Choose a folder containing all the analysed data")
        print("Folder path: " + foldername)
        os.chdir(foldername)
        #Check folder structure to check if all is OK
        folders = os.listdir(foldername)
        bvalid = True #Flag to see if we can continue
        if len(folders) < 2:
            bvalid = False
            print("not enough folders")
        else:
            # print("Folders")
            # print(folders)

            #Check data folder (should be 1st in the folder when sorted alphabetically)
            dfolder = folders[0]
            dfiles = os.listdir(dfolder)
            # print("Data files")
            # print(dfiles)
            os.chdir(foldername + "/" + dfolder)
            dfile = 0 #Placeholder for datafile
            out_file = 0 #Placeholder for OUT file

            #Find the OUT file
            # print("OUT file")
            for file in glob.glob('*OUT.csv'):
                # print(file)
                out_file = file
                #Find the data file (same prefix as OUT file)
                dfile = out_file.split("_OUT")[0] + ".csv"
                if not os.path.exists(dfile):
                    dfile = 0
            if (dfile == 0) or (out_file == 0):
                bvalid = False
                print("data files missing")
            # else:
            #     print("Data files")
            #     print(dfile)
            #     print(out_file)

            #Check video folders
            vfolder = folders[1]
            vfiles = os.listdir(foldername + "/"+ vfolder)
            # print("Video files")
            # print(vfiles)
            os.chdir(foldername + "/" + vfolder)
            avi_file = 0 #Placeholder for AVI file
            # mp4_file = 0 #Placeholder for MP4 file
            vid1 = 0 #placeholder for first video
            #Find the AVI file
            # print("AVI file")
            avi_list = glob.glob('*.mp4')

            if len(avi_list) == 0:
                mov_list = glob.glob('*.mov')
                for file in mov_list:
                    # print(file)
                    avi_file = file
                    #Save the first video file location
                    if vid1 == 0:
                        vid1 = avi_file
                        print("vid file" + vid1)
                    #Find the data file (same prefix as OUT file)
                    mp4_file = avi_file.split(".")[0] + ".mp4"
                    if not os.path.exists(mp4_file):
                        mp4_file = 0
            else:
                for file in avi_list:
                    # print(file)
                    avi_file = file
                    #Save the first video file location
                    if vid1 == 0:
                        vid1 = avi_file
                        print("vid file" + vid1)
                    #Find the data file (same prefix as OUT file)
                    mp4_file = avi_file.split(".")[0] + ".mp4"
                    if not os.path.exists(mp4_file):
                        mp4_file = 0

            if (mp4_file == 0) or (avi_file == 0):
                bvalid = False
                print("video files missing")
            # else:
            #     print("Video files")
            #     print(mp4_file)
            #     print(avi_file)
            #Find the vt file
            vt_file = 0
            for file in glob.glob('*.csv'):
                if file == "vt.csv":
                    vt_file = file
            if vt_file == 0:
                bvalid = False
                print("Video not synched to IMU")

        if bvalid:
            dat.view_only = True #Set to view only mode
            #Disable specific menus
            # dat.filemenu.entryconfigure("Load csv...", state='disabled')
            # dat.filemenu.entryconfigure("Load avi...", state='disabled')
            # dat.filemenu.entryconfigure("Load gps...", state='disabled')
            # dat.filemenu.entryconfigure("Plot data", state='disabled')
            # dat.filemenu.entryconfigure("Export events", state='disabled')
            # dat.filemenu.entryconfigure("Export config", state='disabled')
            # dat.filemenu.entryconfigure("Import config", state='disabled')
            # # dat.filemenu.entryconfigure("Choose axes", state='disabled')
            # dat.filemenu.entryconfigure("Save working directory...", state='disabled')

            dat.filename = foldername + "/"+ dfolder + "/" + dfile
            dat.wd = foldername + "/"+ vfolder+"/"
            dat.vid1 = dat.wd + "/"+ vid1
            #Load and plot data
            #Load standard file
            print("loading data")
            Menu_functions_FILE.load_csv(self,dat)
            dat.temp_done = False
            while(dat.temp_done == False):
                self.update()
            #Load OUT file
            dat.filename = foldername + "/"+ dfolder + "/" + out_file #Change filename to OUT path
            Menu_functions_FILE.load_csv(self,dat)
            dat.filename = foldername + "/"+ dfolder + "/" + dfile #Change the path back
            #Plot the data
            print("plotting data")
            Menu_functions_PLOT.plot_dat(self,dat)
            #Load video
            print("saving wd")
            Menu_functions_VIDEO.save_wd(dat)
            print("loading avi")
            Menu_functions_VIDEO.load_avi(self,dat)

            dat.view_only = False #Comment out this line if view_only should be used

            #Plot audio
            Menu_functions_PLOT.plot_audio(self,dat)
        else:
            print("Data structure not correct")
            showinfo(title='ERROR',message='Folder structure not correct')
    def vid_control(self,dat):
        w3 = tk.Toplevel(self)
        w3.title("Video controls")
        w3.geometry("800x1000")
        w3.attributes('-topmost',True)
        def fclose(self,dat):
            w3.destroy()

        dat.btn_play = tk.Button(w3,text  = "PLAY/PAUSE ( -p- )", command=lambda:Keyboard_functions.vid_play(self,dat))
        dat.btn_play.pack()
        dat.btn_play = tk.Button(w3,text  = "REWIND ( -r- )", command=lambda:Keyboard_functions.vid_rewind(self,dat))
        dat.btn_play.pack()
        dat.btn_fwd = tk.Button(w3,text  = "Frame FWD ->> ( -n- )", command=lambda:Keyboard_functions.frame_fwd(self,dat))
        dat.btn_fwd.pack()
        dat.btn_bkwd = tk.Button(w3,text  = "Frame <<- BKWD ( -b- )", command=lambda:Keyboard_functions.frame_bkwd(self,dat))
        dat.btn_bkwd.pack()
        dat.btn_jump = tk.Button(w3,text  = "Skip video to selected point ( -j- )", command=lambda:Keyboard_functions.jump_frame(self,dat))
        dat.btn_jump.pack()

        btn_close = tk.Button(w3,text = "Close",command = lambda:fclose(self,dat))
        btn_close.pack()
    def analysis_seq(self,dat):
        w4 = tk.Toplevel(self)
        w4.title("Analysis cheatsheet")
        w4.geometry("400x400")
        # w4.attributes('-topmost',True)
        w4.wm_transient(self)


        dat.btn_1 = tk.Button(w4,text = "1a. Load data", command=lambda:Menu_functions_FILE.load_csv(self,dat) )
        dat.btn_1.pack()
        if dat.df_loaded:
            dat.btn_1['state'] = "disabled"
        dat.btn_2 = tk.Button(w4,text = "1b. Load OUT file (optional)", command=lambda:Menu_functions_FILE.load_csv(self,dat) )
        dat.btn_2.pack()
        if dat.out_file_loaded:
            dat.btn_2['state'] = "disabled"
        dat.btn_3 = tk.Button(w4,text = "2. Plot data", command=lambda:Menu_functions_PLOT.plot_dat(self.data_frame,dat))
        dat.btn_3.pack()
        if dat.out_file_loaded:
            dat.btn_3['state'] = "disabled"
        dat.btn_4 = tk.Button(w4,text = "3a. Set Video directory", command=lambda:Menu_functions_VIDEO.save_wd(dat))
        dat.btn_4.pack()
        if dat.out_file_loaded:
            dat.btn_4['state'] = "disabled"
        dat.btn_5 = tk.Button(w4,text = "3b. Load video", command=lambda:Menu_functions_VIDEO.load_avi(self,dat))
        dat.btn_5.pack()
        if dat.out_file_loaded:
            dat.btn_5['state'] = "disabled"
        def fclose(self,dat):
            w4.destroy()
        btn_close = tk.Button(w4,text = "Close",command = lambda:fclose(self,dat))
        btn_close.pack()

class Menu_functions_MODEL(Data_Frame):
    def model_YOLO(self,dat):
        file_types = (('pt files', '*.pt'),('All files', '*.*'))
        yolo_file = fd.askopenfilename(
                    title='Open a file',
                    # initialdir=dat.wd,
                    filetypes=file_types)

        print("LOADING YOLO model.... " + yolo_file)
        try:
            dat.yolo_model = YOLO(yolo_file)
            dat.yolo_loaded = 1
            print("SUCCESSFULLY LOADED YOLO MODEL")
        except:
            print("YOLO model FAILED")

    def run_YOLO(self,dat):

        #Check if the YOLO model is loaded
        if(dat.yolo_loaded ==1):

            #assign columns to the dataframe
            if 'Krill' not in dat.df:
                dat.df = dat.df.assign(Krill = 0)
            if 'Penguin' not in dat.df:
                dat.df = dat.df.assign(Penguin = 0)
            if 'Head' not in dat.df:
                dat.df = dat.df.assign(Head = 0)

            #Load previous run model results if they are present
            #FInd the file name
            pre,ext = os.path.splitext(dat.filename)
            out_file = pre + "_YOLO.csv"
            #Choose the columns
            col_select = ["Krill","Penguin","Head"]

            #Choose the column where the YOLO predictions will be saved
            wtemp = tk.Toplevel(self)
            wtemp.title("Please select the output column for the YOLO predictions")
            wtemp.geometry("600x400")
            wtemp.attributes('-topmost',True)

            #temporary column holder
            temp_col = tk.StringVar()
            temp_col.set('Select column')

            #accX
            ax_menu = tk.OptionMenu(wtemp, temp_col, *col_select)
            tk.Label(wtemp, text="Choose column").grid(row=0, column=0)
            ax_menu.grid(row=0, column =1)

            #Function when the "Apply button is pressed"
            #Once the column name has been chosen we continue
            def tapply():
                dat.yolo_col = temp_col.get()
                print(f'{dat.yolo_col} - Selected')
                dat.temp_done = True
                wtemp.destroy()

            #Create the button and add the
            btnApply = tk.Button(wtemp,text = "Apply",command = tapply)
            btnApply.grid(row = 3, column = 0)
            #Wait for a decision
            dat.temp_done = False
            while dat.temp_done == False:
                self.update()

            #Run though all the video files and predict with the model
            print("Running YOLO model")
            dat.byolo = True #Flag to say we are busy with the yolo model

            ##################################
            #For running the loaded video only
            ##################################

            #Create a new window to show a progress bar and cancel button
            #New temp window
            w2 = tk.Toplevel(self)
            w2.title("YOLO predicting")
            w2.geometry("400x200")
            label = tk.Label(w2, text="Please wait - this may take a while")  # create the label
            label.pack()  # add the label to the window
            def fcancel(self,dat):
                print("YOLO model cancelled")
                dat.byolo = False
                w2.destroy()
            #Place a cancel button
            btn_close = tk.Button(w2,text = "Cancel",command = lambda:fcancel(self,dat))
            btn_close.pack()

            #Initiate the progress bar
            pb = ttk.Progressbar(
             w2,
             orient='horizontal',
             mode='determinate',
             length=350
             # value = 0
             )
            pb.pack()
            # #Loop through all the frames and predict while saving the output to the dataframe
            print("YOLO predicting in progress...")
            frame = 1
            dat.vid.set(1,frame)
            while True:
                 # Read a frame from the video
                 ret, image = dat.vid.read()

                 # print(frame)
                 pb['value'] = (frame/dat.frame_count)*100
                 # print(frame_count)
                 # print((frame/frame_count)*350)
                 w2.update_idletasks()

                 #Update the screen
                 self.update()
                 #Calculate the millisecond time
                 time = float(frame)/dat.fps

                 #Get the time of the frame
                 frame_date = dat.vid_start_date + timedelta(seconds = time)
                 # print(f'Frame date: {frame_date}')
                 idx = (np.abs(dat.df.iloc[:,dat.time_col] - frame_date)).argmin() #Find the row in the df with the nearest value to ipick
                 # print(f'idx: {idx}')
                 if ret:
                     #Predict on the frame using the loaded model
                     if dat.yolo_col == "Penguin":
                         yolo_predict = dat.yolo_model.predict(image,
                         conf = 0.3,
                         iou = 0.6,
                         nms = True,
                         max_det = 50,
                         seed = 42,
                         verbose=False)
                     elif dat.yolo_col == "Krill":
                         yolo_predict = dat.yolo_model.predict(image,
                         conf = 0.5,
                         verbose=False)
                     elif dat.yolo_col == "Head":
                         yolo_predict = dat.yolo_model.predict(image,
                         conf = 0.3,
                         iou = 0.3,
                         nms = True,
                         max_det = 1,
                         seed = 42,
                         verbose=False)

                     #Read the results
                     model_result = yolo_predict[0]

                     #Classifier
                     if (model_result.probs is not None) and (dat.yolo_col == "Krill"):
                         #Extract the results and label
                         cls =  round(model_result.probs.top1,2)
                         if cls == 0:
                             cls = 1
                             cls_name = "Krill"
                             font_color = (255, 12, 12) #Red color text
                             dat.df.iloc[idx,dat.df.columns.get_loc("Krill")] = 1
                         else:
                             cls=0
                             cls_name = "No_Krill"
                             font_color = (36, 255, 12) #Green color text
                             dat.df.iloc[idx,dat.df.columns.get_loc("Krill")] = 0
                     #Object detector
                     elif (model_result.boxes is not None) and (dat.yolo_col == "Penguin" or dat.yolo_col == "Head"):
                             #Find all the bounding boxes
                             bboxes = model_result.boxes.xyxy.int().tolist()
                             # print(bboxes)
                             bboxes2 = model_result.boxes.xywhn.int().tolist()
                             #Extract the model results
                             cls =  model_result.boxes.cls.int().tolist()
                             #Extract the probabilities
                             confs =  model_result.boxes.conf.tolist()
                             #Mergee the boxes, labels and probs
                             bb_cls_cf = [(bb,cl,cf) for cl,bb,cf in zip(cls,bboxes,confs)]
                             #Draw the boxes on the image
                             box_count = 0
                             for  bb,cl,cf in bb_cls_cf:
                                 b_x1,b_y1,b_x2,b_y2 =  bb

                                 cv2.rectangle(image,(b_x1,b_y1),(b_x2,b_y2), (0, 255, 0), 2)
                                 cv2.putText(image, str(cf), (b_x1, b_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                 box_count+=1
                             if dat.yolo_col == "Penguin":
                                 dat.df.iloc[idx,dat.df.columns.get_loc("Penguin")] = box_count
                             elif dat.yolo_col == "Head":
                                 dat.df.iloc[idx,dat.df.columns.get_loc("Head")] = 1-round(b_y1/image.shape[0],2)
                                 # print(1-round(b_y1/image.shape[0],2))

                 if (frame >= dat.frame_count)  or (dat.byolo == False):
                     print("end of video")
                     w2.destroy()
                     break
                 else:
                     frame += 1


            #For running all files
            # v_num = -1       #video location in the folder
            # files = sorted(os.listdir(dat.wd)) #All files within the video directory
            # # print(files)
            # #Loop through all the video files
            # while (v_num < len(files)) and dat.byolo == True:
            #         v_num = v_num + 1 #Increment the file location
            #         #Look for the next mp4 file
            #         while(((not files[v_num].lower().endswith(".mp4")))  and v_num < (len(files)-1)):
            #             v_num = v_num + 1
            #         #Stop if no more mp4 files are present
            #         if not files[v_num].lower().endswith(".mp4"):
            #             break
            #         filename = dat.wd +'/' + files[v_num]
            #         print(filename)
            #         #Squash image
            #         if dat.byolo == True:
            #            #Load the video
            #            vid = cv2.VideoCapture(filename)
            #            #Get the frame rate
            #            fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
            #            #Get the frame count
            #            frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            #            #Video Duration
            #            dur = frame_count*fps
            #
            #            #Get the video name (without suffix)
            #            pre,ext = os.path.splitext(files[v_num])
            #            frame = 0
            #
            #            vid_match = [video[:-4] for video in dat.vt["vid"]].index(pre)
            #            print(vid_match)
            #            #Read the video start date-time from the vt file
            #            if(int(dat.vt["set"][vid_match]) == 1):
            #                vid_start_date = dat.vt["vid_start_date"][vid_match]
            #                video_offset = vid_start_date - dat.vt["Timestamp"][vid_match]#
            #            else:
            #                dat.byolo = 0
            #                print("Video time is not synched - can't continue. Please synch the time and try again")
            #                break
            #
            #            #Create a new window to show a progress bar and cancel button
            #            #New temp window
            #            w2 = tk.Toplevel(self)
            #            w2.title("YOLO predicting")
            #            w2.geometry("400x200")
            #            label = tk.Label(w2, text="Please wait - this may take a while")  # create the label
            #            label.pack()  # add the label to the window
            #            def fcancel(self,dat):
            #                print("YOLO model cancelled")
            #                dat.byolo = False
            #                w2.destroy()
            #            #Place a cancel button
            #            btn_close = tk.Button(w2,text = "Cancel",command = lambda:fcancel(self,dat))
            #            btn_close.pack()
            #
            #            #Initiate the progress bar
            #            pb = ttk.Progressbar(
            #             w2,
            #             orient='horizontal',
            #             mode='determinate',
            #             length=350
            #             # value = 0
            #             )
            #            pb.pack()
            #            # #Loop through all the frames and predict while saving the output to the dataframe
            #            print("YOLO predicting in progress...")
            #            while True:
            #                 # Read a frame from the video
            #                 ret, image = vid.read()
            #
            #                 #Increase the frame counter
            #                 frame += 1
            #                 # print(frame)
            #                 pb['value'] = (frame/frame_count)*350
            #                 # print(frame_count)
            #                 # print((frame/frame_count)*350)
            #                 w2.update_idletasks()
            #
            #                 #Update the screen
            #                 self.update()
            #                 #Calculate the millisecond time
            #                 time = float(frame)/fps
            #
            #                 #Get the time of the frame
            #                 frame_date = vid_start_date + timedelta(seconds = time)
            #                 print(f'Frame date: {frame_date}')
            #                 idx = (np.abs(dat.df.iloc[:,dat.time_col] - frame_date)).argmin() #Find the row in the df with the nearest value to ipick
            #                 print(f'idx: {idx}')
            #                 if ret:
            #                     #Predict on the frame using the loaded model
            #                     yolo_predict = dat.yolo_model.predict(image, verbose=False)
            #                     #Read the results
            #                     model_result = yolo_predict[0]
            #
            #                     #Classifier
            #                     if model_result.probs is not None:
            #                         #Extract the results and label
            #                         cls =  round(model_result.probs.top1,2)
            #                         if cls == 0:
            #                             cls = 1
            #                             cls_name = "Krill"
            #                             font_color = (255, 12, 12) #Red color text
            #                             dat.df.iloc[idx,dat.df.columns.get_loc("Krill")] = 1
            #                         else:
            #                             cls=0
            #                             cls_name = "No_Krill"
            #                             font_color = (36, 255, 12) #Green color text
            #                             dat.df.iloc[idx,dat.df.columns.get_loc("Krill")] = 0
            #                         # #probability values
            #                         # confs =  round(model_result.probs.top1conf.item(),2)
            #                         # #Text to be printed on image
            #                         # text = f'Class: {cls_name}, Conf: {confs}'
            #                         # font = cv2.FONT_HERSHEY_SIMPLEX
            #                         # font_scale = 0.9
            #                         # font_thickness = 2
            #                         #
            #                         # # Get text size
            #                         # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            #                         #
            #                         # # Calculate the starting point for the text (bottom-right corner)
            #                         # image_height, image_width = image.shape[:2]
            #                         # text_position = (image_width - text_size[0] - 10, image_height - 10)
            #                         #
            #                         # # Draw the text on the image
            #                         # cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)
            #
            #                     #Object detector
            #                     elif model_result.boxes is not None:
            #                         #Find all the bounding boxes
            #                         bboxes = model_result.boxes.xyxy.int().tolist()
            #                         #Extract the model results
            #                         cls =  model_result.boxes.cls.int().tolist()
            #                         #Extract the probabilities
            #                         confs =  model_result.boxes.conf.tolist()
            #                         #Mergee the boxes, labels and probs
            #                         bb_cls_cf = [(bb,cl,cf) for cl,bb,cf in zip(cls,bboxes,confs)]
            #                         #Draw the boxes on the image
            #                         box_count = 0
            #                         for  bb,cl,cf in bb_cls_cf:
            #                             b_x1,b_y1,b_x2,b_y2 =  bb
            #                             cv2.rectangle(image,(b_x1,b_y1),(b_x2,b_y2), (0, 255, 0), 2)
            #                             cv2.putText(image, str(cf), (b_x1, b_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            #                             box_count+=1
            #                         dat.df.iloc[idx,dat.df.columns.get_loc("Penguin")] = box_count
            #                     else:
            #                         showinfo("Please load your preferred YOLO model first")
            #                 if (frame >= frame_count)  or (dat.byolo == False):
            #                     print("end of video")
            #                     w2.destroy()
            #                     break

            #Save the output
            if dat.view_only == False:
                dat.df.loc[:,col_select].to_csv(out_file,index = False)
                print("YOLO SAVED")
        else:
         showinfo("Please load your preferred YOLO model first")

class Button_functions(Data_Frame):
    #Normalization of plotting axes
    ##!! REMOVING THIS FOR THE TIME BEING AS IT TAKES UP TOO MUCH MEMORY!!
    # def normalize(self):
    #     if(self.norm.get() == 1):
    #         df_hold = copy.deepcopy(self.df_hold)
    #         for i in self.df.columns:
    #             if self.df[i].dtypes == "float64":
    #                 self.df[i] = df_hold[i].rolling(self.smooth,center = True).mean()
    #                 fmin = (df_hold[i].min())
    #                 fmax = (df_hold[i].max())
    #                 self.df[i] = 2*(df_hold[i] - fmin)/(fmax-fmin)*1 -1
    #
    #     elif(self.norm.get() == 0):
    #         df_hold = copy.deepcopy(self.df_hold)
    #         for i in self.df.columns:
    #             if self.df[i].dtypes == "float64":
    #                 self.df[i] = df_hold[i]
    def next_vid(self,dat):
    ##        print(dat.vid1)
    ##
    ##        try:
    ##            temp = re.search(r"\((\d+)\)", dat.vid1)
    ##            print(temp)
    ##        except:
    ##            v_num  = int(re.search(r"\0(\w+)\.", dat.vid1).group(1))
    ##
    ##        v_num  = int(temp.group(1))
    ##        print("v_num " + str(v_num))
    ##        wd =
        files = sorted(os.listdir(dat.wd))

        if(dat.v_num < len(files)):
            # print(files[dat.v_num])
            dat.v_num = dat.v_num + 1
            # print(files[dat.v_num])
            while(((not files[dat.v_num].lower().endswith(".mp4")))  and dat.v_num < len(files)):
                dat.v_num = dat.v_num + 1
            # print(dat.v_num)
            print(files[dat.v_num])
            dat.vid1 = dat.wd + "/" + files[dat.v_num]
            Menu_functions_VIDEO.load_avi(self,dat,dat.vid1)
    #
    #
    #
    #         pre,ext = os.path.splitext(dat.vid1)
    # ##            dat.vid1 =  pre + ".avi"
    # ##            print(dat.vid1)
    # ##        dat.vid1 = filename
    # ##        showinfo(title='Selected File',message=filename)
    #
    #         dat.vid = cv2.VideoCapture(dat.vid1)
    #         dat.frame = 0
    #         dat.frame_count = int(dat.vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #         dat.fps = int(dat.vid.get(cv2.CAP_PROP_FPS))
    #
    #
    #         #Get the timestamp time from the video metadata
    #         #First - Run ffprobe command to get video metadata
    #         ffprobe_command = [
    #             'ffprobe',
    #             '-v', 'quiet',
    #             '-print_format', 'json',
    #             '-show_format',
    #             dat.vid1
    #         ]
    #
    #         try:
    #             ffprobe_output = subprocess.check_output(ffprobe_command, stderr=subprocess.STDOUT)
    #             metadata = json.loads(ffprobe_output.decode('utf-8'))
    #             if 'creation_time' in metadata['format']['tags']:
    #                 creation_time = metadata['format']['tags']['creation_time']
    #                 dat.vid_last_date = pd.to_datetime(creation_time,utc = utc).timestamp()# - timedelta(seconds = dat.frame_count/dat.fps)
    #             else:
    #                 print('no creation time')
    #                 #Otherwise check the creation date using os.path
    #                 dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
    #         except subprocess.CalledProcessError as e:
    #             print("Error:", e)
    #         # pre,ext = os.path.splitext(dat.vid1)
    #         # dat.vid1 =  pre + ".mp4"
    #         # print(dat.vid1)
    #         # dat.vid = cv2.VideoCapture(dat.vid1)
    #
    #         # print(dat.fps)
    #
    #         files = sorted(os.listdir(dat.wd))
    #         video_name = files[dat.v_num]
    #         vn =  os.path.splitext(video_name)[0]
    #
    #         vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)
    #         if(int(dat.vt["set"][vid_match]) == 1): #if(int(dat.vt["set"][vid_match]) == 1):
    #             dat.vid_date_set = 1
    #             # print(pd.to_datetime(dat.vt["vid_start_date"][vid_match],utc = utc))
    #             dat.vid_start_date = dat.vt["vid_start_date"][vid_match]
    #             dat.video_offset = dat.vid_start_date - dat.vt["Timestamp"][vid_match] #timedelta(seconds = int(dat.vt["video_offset"][vid_match]))
    #         else:
    #             dat.vid_date_set = 0
    #             # dat.vid_start_date = pd.to_datetime(dat.vid_last_date -(dat.frame_count/dat.fps),unit = 's',utc = utc)
    #             dat.vid_start_date = pd.to_datetime(dat.vid_last_date ,unit = 's',utc = utc)
    #             print("original date: " + str(dat.vid_start_date))
    #             vid_start_date = dat.vid_start_date + dat.video_offset
    #             print("offset: " + str(dat.video_offset))
    #
    #             dat.vid_start_date = vid_start_date
    #             print("adjusted date: " + str(dat.vid_start_date))
    #
    #         dat.frame_date=dat.vid_start_date
    #
    #         dat.frame = 1
    #         dat.vid.set(1,dat.frame)
    #         s, dat.image = dat.vid.read()
    #         dat.video_loaded = True
    #         dat.image = cv2.cvtColor(dat.image, cv2.COLOR_BGR2RGB)
    #         dat.image = cv2.resize(dat.image,(dat.video_frame.winfo_width(),int(dat.video_frame.winfo_width()/1.5)))
    #         dat.image = Image.fromarray(dat.image)
    #         dat.image = ImageTk.PhotoImage(dat.image)
    #         dat.panel = tk.Label(dat.video_frame,image=dat.image)
    #         dat.panel.image = dat.image
    #         dat.panel.grid(column = 0, row = 0,sticky = 'nesw')
    # #             dat.panel.pack()
    # # ##            dat.panel.place(x = 700,y = 90)
    # #             dat.panel.place(relx = 0.5,rely = 0.1)
    #
    #         ########################################
    #         #Add navigation bar and squashed image
    #         ########################################
    #
    #         #Nav Bar
    #         # define the callback function for the trackbar
    #         def on_trackbar(ival):
    #             # print(ival)
    #             dat.frame = int(ival)
    #             dat.vid.set(1,int(ival))
    #             Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()
    #
    #         dat.tbar = tk.Scale(from_=0, to=dat.vid.get(cv2.CAP_PROP_FRAME_COUNT)-1, orient=tk.HORIZONTAL, command=on_trackbar, showvalue= 0,length = dat.video_frame.winfo_width())
    #         dat.tbar.place(in_=dat.panel ,relx = 0,rely = 1  )
    #
    #         #Squashed Img
    #         try:
    #             dat.sq_panel.destroy()
    #         except:
    #             pass
    #         squash_file = pre + "_squash.jpg"
    #         if os.path.exists(squash_file):
    #             dat.sq_img = cv2.imread(squash_file)
    #             if dat.sq_img.shape[0] > dat.sq_img.shape[1]: #height > width
    #                 dat.sq_img =cv2.rotate(dat.sq_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #             dat.sq_img_re = cv2.resize(dat.sq_img,(dat.video_frame.winfo_width(),50))
    #             dat.sq_img_re = Image.fromarray(dat.sq_img_re)
    #             dat.sq_img_re = ImageTk.PhotoImage(dat.sq_img_re)
    #             # dat.sq_panel = tk.Button(image=dat.sq_img,command = dat.frame_from_squash(event,self))#,width = dat.video_width,height = 10)
    #             dat.sq_panel = tk.Label(image=dat.sq_img_re)
    #             dat.sq_panel.image = dat.sq_img_re
    #             dat.sq_panel.place(in_ = dat.tbar, relx = 0, rely = 1)
    #             #Add a function to the image
    #             dat.sq_panel.bind("<Button-1>",lambda event:Button_functions.frame_from_squash(event,self))
    #
    # ##        dat.panel.bind("<KeyPress>", lambda x: print(x.char))
    #         if(dat.date_set == 1 and dat.df_loaded == True ):
    #             dat.ax_main.cla()
    #
    #             dat.vid_idx_start = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin()
    #             dat.sub_min = dat.vid_idx_start
    #
    #             dat.vid_idx_end = dat.vid_idx_start + int((dat.frame_count/dat.fps)*dat.frequency)
    #             dat.sub_max = dat.sub_min + 1000
    #
    #             i_int = 1000
    #             if((dat.vid_idx_end - dat.vid_idx_start) < 1000):
    #                 i_int = 10
    #
    #             dat.ax_main.plot(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.time_col],dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.p_cols])
    #             dat.temp_vline1 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_start,dat.time_col],color = 'blue')
    #             dat.temp_vline2 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_end,dat.time_col],color = 'blue')
    #             dat.figure_main.canvas.draw()
    #
    #             dat.ax_zoom.cla() #Clear axes
    #             dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
    #             dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    #             # dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
    #             dat.figure_zoom.canvas.draw() #Redraw the figure
    #             self.update()
    #         #Add button to synchronize time
    #         Menu_functions_FILE.add_synch_button(self,dat)
    #
    #         ########################################
    #         #Add sound if available
    #         ########################################
    #         # try:
    #         if os.path.exists(pre+".wav"):
    #             dat.audio_present = True
    #             input_data = read(pre+".wav")
    #             dat.audio = pd.DataFrame()
    #             dat.audio['aud'] = input_data[1]
    #
    #             start_time = dat.vid_start_date
    #
    #             total_seconds = dat.frame_count/dat.fps
    #             time_interval = timedelta(seconds=(total_seconds / (len(dat.audio) - 1)))
    #
    #             # Generate time axis
    #             time_axis = [start_time + i * time_interval for i in range(len(dat.audio))]
    #             dat.audio[dat.time_col_string] = time_axis
    #
    #             print("Audio loaded")
    #         else:
    #             dat.audio_present = False
    #         # except:
    #         #     print("no audio found")
    def prev_vid(self,dat):

        files = sorted(os.listdir(dat.wd))
        # print(files[dat.v_num])
        dat.v_num = dat.v_num - 1
        # print(files[dat.v_num])
        if(dat.v_num < 0):
            print("video file error")
        else:
            while(((not files[dat.v_num].lower().endswith(".mp4")) and (not files[dat.v_num].lower().endswith(".mov")))  and dat.v_num >= 0):
                dat.v_num = dat.v_num - 1
            print(files[dat.v_num])

            dat.vid1 = dat.wd + "/" +files[dat.v_num]

            Menu_functions_VIDEO.load_avi(self,dat,dat.vid1)

    #         dat.vid = cv2.VideoCapture(dat.vid1)
    #         dat.frame = 0
    #         dat.frame_count = int(dat.vid.get(cv2.CAP_PROP_FRAME_COUNT))
    #         dat.fps = int(dat.vid.get(cv2.CAP_PROP_FPS))
    #
    #         #Get the timestamp time from the video metadata
    #         #First - Run ffprobe command to get video metadata
    #         ffprobe_command = [
    #             'ffprobe',
    #             '-v', 'quiet',
    #             '-print_format', 'json',
    #             '-show_format',
    #             dat.vid1
    #         ]
    #
    #         try:
    #             ffprobe_output = subprocess.check_output(ffprobe_command, stderr=subprocess.STDOUT)
    #             metadata = json.loads(ffprobe_output.decode('utf-8'))
    #             if 'creation_time' in metadata['format']['tags']:
    #                 creation_time = metadata['format']['tags']['creation_time']
    #                 dat.vid_last_date = pd.to_datetime(creation_time,utc = utc).timestamp()# - timedelta(seconds = dat.frame_count/dat.fps)
    #             else:
    #                 print('no creation time')
    #                 #Otherwise check the creation date using os.path
    #                 dat.vid_last_date = os.path.getmtime(dat.vid1) #Read file creation date (this is the date at the end of the clip) and covert to POSIX
    #         except subprocess.CalledProcessError as e:
    #             print("Error:", e)
    #
    #
    #         pre,ext = os.path.splitext(dat.vid1)
    #         # dat.vid1 =  pre + ".mp4"
    #         # print(dat.vid1)
    #         # dat.vid = cv2.VideoCapture(dat.vid1)
    #
    #         # print(dat.fps)
    #         files = sorted(os.listdir(dat.wd))
    #         video_name = files[dat.v_num]
    #         vn =  os.path.splitext(video_name)[0]
    #         #
    #         vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)
    #         if(int(dat.vt["set"][vid_match]) == 1): #if(int(dat.vt["set"][vid_match]) == 1):
    #             dat.vid_date_set = 1
    #             dat.vid_start_date = dat.vt["vid_start_date"][vid_match]
    #             dat.video_offset = dat.vid_start_date - dat.vt["Timestamp"][vid_match] #timedelta(seconds = int(dat.vt["video_offset"][vid_match]))
    #         else:
    #             dat.vid_date_set = 0
    #             # dat.vid_start_date = pd.to_datetime(dat.vid_last_date -(dat.frame_count/dat.fps),unit = 's',utc = utc)
    #             dat.vid_start_date = pd.to_datetime(dat.vid_last_date ,unit = 's',utc = utc)
    #             vid_start_date = dat.vid_start_date + dat.video_offset
    #             dat.vid_start_date = vid_start_date
    #         dat.frame_date=dat.vid_start_date
    #
    #         dat.frame = 1
    #         dat.vid.set(1,dat.frame)
    #         s, dat.image = dat.vid.read()
    #         dat.video_loaded = True
    #         dat.image = cv2.cvtColor(dat.image, cv2.COLOR_BGR2RGB)
    #         dat.image = cv2.resize(dat.image,(dat.video_frame.winfo_width(),int(dat.video_frame.winfo_width()/1.5)))
    #         dat.image = Image.fromarray(dat.image)
    #         dat.image = ImageTk.PhotoImage(dat.image)
    #         dat.panel = tk.Label(dat.video_frame,image=dat.image)
    #         dat.panel.image = dat.image
    #         dat.panel.grid(column = 0, row = 0,sticky = 'nesw')
    # #             dat.panel.pack()
    # # ##            dat.panel.place(x = 700,y = 90)
    # #             dat.panel.place(relx = 0.5,rely = 0.1)
    # #     ##        dat.panel.bind("<KeyPress>", lambda x: print(x.char))
    #
    #         ########################################
    #         #Add navigation bar and squashed image
    #         ########################################
    #
    #         #Nav Bar
    #         # define the callback function for the trackbar
    #         def on_trackbar(ival):
    #             # print(ival)
    #             dat.frame = int(ival)
    #             dat.vid.set(1,int(ival))
    #             Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()
    #
    #         dat.tbar = tk.Scale(from_=0, to=dat.vid.get(cv2.CAP_PROP_FRAME_COUNT)-1, orient=tk.HORIZONTAL, command=on_trackbar, showvalue= 0,length = dat.video_frame.winfo_width())
    #         dat.tbar.place(in_=dat.panel ,relx = 0,rely = 1  )
    #
    #         #Squashed Img
    #         try:
    #             dat.sq_panel.destroy()
    #         except:
    #             pass
    #         squash_file = pre + "_squash.jpg"
    #         if os.path.exists(squash_file):
    #             dat.sq_img = cv2.imread(squash_file)
    #             if dat.sq_img.shape[0] > dat.sq_img.shape[1]: #height > width
    #                 dat.sq_img =cv2.rotate(dat.sq_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #             dat.sq_img_re = cv2.resize(dat.sq_img,(dat.video_frame.winfo_width(),50))
    #             dat.sq_img_re = Image.fromarray(dat.sq_img_re)
    #             dat.sq_img_re = ImageTk.PhotoImage(dat.sq_img_re)
    #             # dat.sq_panel = tk.Button(image=dat.sq_img,command = dat.frame_from_squash(event,self))#,width = dat.video_width,height = 10)
    #             dat.sq_panel = tk.Label(image=dat.sq_img_re)
    #             dat.sq_panel.image = dat.sq_img_re
    #             dat.sq_panel.place(in_ = dat.tbar, relx = 0, rely = 1)
    #             #Add a function to the image
    #             dat.sq_panel.bind("<Button-1>",lambda event:Button_functions.frame_from_squash(event,self))
    #
    #         if(dat.date_set == 1 and dat.df_loaded == True):
    #             dat.ax_main.cla()
    #             # print(dat.vid_start_date)
    #             # print(dat.df.iloc[0,dat.time_col])
    #             # print(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)
    #             dat.vid_idx_start = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin()
    #             dat.sub_min = dat.vid_idx_start
    #
    #
    #             # print(dat.vid_idx_start)
    #             # print(dat.df.iloc[dat.vid_idx_start,dat.time_col])
    #             dat.vid_idx_end = dat.vid_idx_start + int((dat.frame_count/dat.fps)*dat.frequency)
    #             dat.sub_max = dat.sub_min + 1000
    #
    #
    #             # print(dat.vid_idx_end)
    #             # print(dat.df.iloc[dat.vid_idx_end,dat.time_col])
    #             i_int = 1000
    #             if((dat.vid_idx_end - dat.vid_idx_start) < 1000):
    #                 i_int = 10
    #
    #             dat.ax_main.plot(dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.time_col],dat.df.iloc[np.arange(dat.vid_idx_start,(dat.vid_idx_end),i_int),dat.p_cols])
    #             dat.temp_vline1 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_start,dat.time_col],color = 'blue')
    #             dat.temp_vline2 =  dat.ax_main.axvline(x = dat.df.iloc[dat.vid_idx_end,dat.time_col],color = 'blue')
    #             dat.figure_main.canvas.draw()
    #
    #             dat.ax_zoom.cla() #Clear axes
    #             dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
    #             dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
    #             # dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
    #             dat.figure_zoom.canvas.draw() #Redraw the figure
    #             self.update()
    #
    #         #Add button to synchronize time
    #         Menu_functions_FILE.add_synch_button(self,dat)
    #         # dat.btn_SYNC = tk.Button(self,text = "SYNC", command=lambda:Button_functions.set_time(dat))
    #         # dat.btn_SYNC.place(in_=dat.panel ,relx = 0,rely=  0 )
    #         # if dat.date_set:
    #         #     dat.btn_SYNC.configure(bg = 'green')
    #         # else:
    #         #     dat.btn_SYNC.configure(bg = 'red')
    #         ########################################
    #         #Add sound if available
    #         ########################################
    #         # try:
    #         if os.path.exists(pre+".wav"):
    #             dat.audio_present = True
    #             input_data = read(pre+".wav")
    #             dat.audio = pd.DataFrame()
    #             dat.audio['aud'] = input_data[1]
    #
    #             start_time = dat.vid_start_date
    #
    #             total_seconds = dat.frame_count/dat.fps
    #             time_interval = timedelta(seconds=(total_seconds / (len(dat.audio) - 1)))
    #
    #             # Generate time axis
    #             time_axis = [start_time + i * time_interval for i in range(len(dat.audio))]
    #             dat.audio[dat.time_col_string] = time_axis
    #
    #             print("Audio loaded")
    #         else:
    #             dat.audio_present = False
    #         # except:
    #         #     print("no audio found")
    def rst_video(self):
        self.frame = 0
    def save_image(self):
        s, self.image = self.vid.read()
        filename = str(self.vid1) + "_" + str(self.frame) + ".jpg"
        cv2.imwrite(filename, self.image)
    def save_vid_clip(self):
        #Saving video clip between selected lines
        frame_start = self.df.iloc[self.dive_min,self.df.columns.get_loc("frame")] #Marked start point
        # print(frame_start)
        frame_end = self.df.iloc[self.dive_max,self.df.columns.get_loc("frame")] #Marked end point
        # print(frame_end)
        # print(int(frame_end) - int(frame_start))
        res = tk.messagebox.askyesno("Export video clip","COntinue?")
        if res:

            size = (int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) )

            # Below VideoWriter object will create
            # a frame of above defined The output
            # is stored in 'filename.avi' file.
            filename = str(self.vid1) + "_" + str(self.frame) + ".mp4"
            result = cv2.VideoWriter(filename,
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     self.fps, size,True)
            self.frame = frame_start
            while (self.frame < frame_end):
    ##                    print(dat.frame)
                self.frame = self.frame + 1
                self.vid.set(1,self.frame)
                s, vframe = self.vid.read()
                # vframe = self.vid.read()
                result.write(vframe)
                self.plot_frame() #Plot the selected image
                # self.update()
            result.release()
            print("Video saved")
    def update_frame(self):
        self.vid.set(1,self.frame)
        self.plot_frame()
    def set_res(self):
        self.video_width = self.res_slider.get()

    #Smoothing output of plotting data
    ##!! REMOVING THIS FOR THE TIME BEING AS IT TAKES UP TOO MUCH MEMORY!!
#     def apply_smooth(self):
#         self.smooth = self.slider2.get()
#         df_hold = copy.deepcopy(self.df_hold)
# ##        print(df_hold.iloc[2,2])
#         print(self.df_hold.iloc[50,2])
#         print(self.smooth)
#         for i in self.df.columns:
#             if self.df[i].dtypes == "float64":
# ##                print(i)
#                 if(self.smooth == 1):
# ##                    pass
#                     self.df[i] = df_hold[i]
#                 else:
# ##                    pass
#                     self.df[i] = df_hold[i].rolling(self.smooth,center = True).mean()
# ##                    self.df[i] = df_hold[i] + self.smooth
# ##        print(df_hold.iloc[50,2])
#         print(self.df_hold.iloc[50,2])

    def frame_from_squash(event,self):
        #Function to get frame number from squashed image
        sq_x = event.x #X coordinate on label (this has to be resized to original)

        #Get resized proportion of original image
        img_prop = self.video_width/self.sq_img.shape[1]

        #Get the frame number and navigate to that point in the video
        new_frame = int(sq_x/img_prop)
        print(f"Squash frame: {new_frame}")

        self.frame = new_frame
        self.vid.set(1,new_frame)
        self.plot_frame()

    def ann1(self,dat,group = False):
        dat.key = 1
        if not group:
            Button_functions.annotate(self,dat)
    def ann2(self,dat,group = False):
        dat.key = 2
        if not group:
            Button_functions.annotate(self,dat)
    def ann5(self,dat,group = False):
        dat.key = 5
        if not group:
            Button_functions.annotate(self,dat)
    def ann6(self,dat,group = False):
        dat.key = 6
        if not group:
            Button_functions.annotate(self,dat)
    def ann7(self,dat,group = False):
        dat.key = 7
        if not group:
            Button_functions.annotate(self,dat)
    def ann9(self,dat,group = False):
        dat.key = 9
        if not group:
            Button_functions.annotate(self,dat)
    def ann0(self,dat,group = False):
        dat.key = 0
        if not group:
            Button_functions.annotate(self,dat)
    def annGroup(self,dat):
        showinfo("Annotate Selection","Mark the area you want to annotate (Start - left click; End - right click)")
        wtemp = tk.Toplevel(self)
        wtemp.title("Enter label")
        wtemp.geometry("200x200")
        # wtemp.attributes('-topmost',True)

        tk.Label(wtemp, text="Annotation").pack()
        entry = tk.Entry(wtemp, width = 25)
        entry.pack()
        def get_entry(self,dat):
            #Check to make sure annotations are numerical
            dat.key = entry.get()
            # print(dat.key)
            Button_functions.annotate(self,dat,True)
            wtemp.destroy()
        tk.Button(wtemp,text = "Annotate", command = lambda:get_entry(self,dat)).pack()
    def set_time(dat):
        res = tk.messagebox.askyesno("Time SYNC","You are about to sync to time to the chosen point. Proceed?")
        if res:
            dat.date_set = 1
            dat.btn_SYNC.configure(bg = 'green')
            dat.bexported = False #Reset the PCE export flag
            print('synch video and IMU')

            # Find vid start date in IMU data
            files = sorted(os.listdir(dat.wd))
            video_name = files[dat.v_num]
            print("video name: " + video_name)
            #First we find any data that might have been saved for this video and remove it
            #This step is very important if adjustments are made to the time synch
            #Find rows that match the video name
            v_rows = dat.df[dat.df['vid'] == video_name].index
            dat.df.loc[v_rows,["BEHAV"]] = 0
            dat.df.loc[v_rows,["vid","frame","vid_time"]] = ""

            dur = dat.frame_count/dat.fps
            print(f"Duration: {dur}")

            idx_start1 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
            idx_end1 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick

            #Save any previous annotations
            pce_hold = dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc(dat.current_label_col)]
            #Clear the axes of any previous data
            dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc(dat.current_label_col)] = 0
            dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc("vid")] = ""
            dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc("frame")] = ""
            dat.df.iloc[np.arange(idx_start1,idx_end1),dat.df.columns.get_loc("vid_time")] = ""

            offset_hold = dat.video_offset
            print("holding offset: " + str(offset_hold))
            dat.video_offset = dat.IMU_date - dat.frame_date
            print("new offset: " + str(dat.video_offset))
            vid_start_date = dat.vid_start_date + dat.video_offset
            print(f"Video start date: {dat.vid_start_date}")
            dat.vid_start_date = vid_start_date
            dat.video_offset = offset_hold + dat.video_offset
            print(f"Corrected start date: {dat.vid_start_date}")

            # print("adjusted offset: " + str(dat.video_offset))
    ##                print(dat.video_offset)
            # print(vid_start_date)
            # print(dat.vid_start_date)

            # # Find vid start date in IMU data
            # files = sorted(os.listdir(dat.wd))
            # video_name = files[dat.v_num]
            # print("video name: " + video_name)

            # dat.vt.iloc[dat.vt["vid"].get_loc(dat.video_name),2] = dat.vid_start_date
            vn =  os.path.splitext(video_name)[0]
            vid_match = [video[:-4] for video in dat.vt["vid"]].index(vn)
            dat.vt["vid_start_date"][vid_match] = dat.vid_start_date
            # print(dat.vid_start_date)
            dat.vt["Timestamp"][vid_match] = dat.vid_start_date - dat.video_offset
            # print(dat.vid_start_date - dat.video_offset)
            dat.vt["video_offset"][vid_match] = dat.video_offset.total_seconds()
            # print(dat.video_offset.total_seconds())
            dat.vt["set"][vid_match] = 1

            vt = dat.vt
            if dat.view_only == False:
                vt.to_csv(dat.wd + "/vt.csv",index = False)
            else:
                print("View only mode - vt not saved")

            # dur = dat.frame_count/dat.fps
            idx_start2 = (np.abs(dat.df.iloc[:,dat.time_col] - dat.vid_start_date)).argmin() #Find the row in the df with the nearest value to ipick
            # print(dat.df.iloc[idx_start2,dat.time_col])
            idx_end2 =  (np.abs(dat.df.iloc[:,dat.time_col] - (dat.vid_start_date + timedelta(seconds = dur)))).argmin() #Find the row in the df with the nearest value to ipick
            # print(dat.df.iloc[idx_end2,dat.time_col])
            # dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid")] = video_name#dat.v_num
            dat.df.loc[np.arange(idx_start2,idx_end2),"vid"] = video_name#dat.v_num

            IMU_range = idx_end2 - idx_start2
            vid_match = np.linspace(0,dat.frame_count,IMU_range)
            dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("frame")] = vid_match

            dat.df.iloc[np.arange(idx_start2,idx_end2),dat.df.columns.get_loc("vid_time")] = dat.df.iloc[np.arange(idx_start2,idx_end2),dat.time_col].dt.floor('s') - dat.video_offset

            dat.df.iloc[np.arange(idx_start2,idx_start2 + (pce_hold.shape[0])),dat.df.columns.get_loc(dat.current_label_col)] = pce_hold

            # for i in range(dat.frame_count):
            #     print(i)
            #     time = float(i)/dat.fps
            #     frame_date = dat.vid_start_date + timedelta(seconds = time)
            #     idx = (np.abs(dat.df.iloc[:,dat.time_col] - frame_date)).argmin() #Find the row in the df with the nearest value to ipick
            #     dat.df["vid"].iloc[idx] = dat.v_num
            #     dat.df["frame"].iloc[idx] = i
    ##            elif key == 'e':
    ##                idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick
    ##                dat.df.iloc[idx,16] = key
    ##                print('mark PCE')
        else:
            pass
    def annotate(self,dat,group = False):
        # if group == True:
        if dat.annotate_selection.get():
            dat.bexported = False #Reset the PCE export flag
            dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc(dat.current_label_col)] = int(dat.key)
            # print(dat.key)
        else:
            idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick
            dat.bexported = False #Reset the PCE export flag
            if dat.key == 1:
                # dat.df.iloc[np.arange((idx-4),(idx + 4),1),16] = 2
                dat.df.iloc[idx,dat.df.columns.get_loc(dat.current_label_col)] = 1
            else:
                dat.df.iloc[idx,dat.df.columns.get_loc(dat.current_label_col)] = int(dat.key)
            # print(dat.key)
        # dat.df['PCE'] = pd.to_numeric(dat.df['PCE'])

class Keyboard_functions(Data_Frame):
    def key_press(event,self,dat):
        # global img_edge,img_blur,img_th,img_bright
        # key = event.char
        key = event.keysym

        # print(key)
        if dat.video_loaded == False and key != "z" and key != "x":
            print("please load video first")
        elif dat.date_set == 0 and key != 's' and key != "space" and key != "r" and key != "right" and key != "n" and key != "b" and key !="left" and key != "z" and key != "x" and key != "shift":
            # print(key)
            print("please synch the video first")
        else:
            if key == "shift":
                print("press left mouse button to mark point for SYNC")
            if key == "j":
                "jump to selected frame"
##                print(dat.frame)
                dat.frame = dat.frame + (dat.IMU_date - dat.frame_date).total_seconds()*dat.fps
##                print(dat.frame)
#             elif key == "c":
#                 dat.df = dat.df.iloc[np.arange(dat.sub_min,dat.sub_max),:]
#                 dat.ax_main.cla() #Clear axes
#                 dat.ax_main.plot(dat.df.iloc[np.arange(0,(dat.nrow),dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(0,(dat.nrow),dat.zoom_int),dat.p_cols]) #Plot new values
# ##                dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
#                 dat.figure_main.canvas.draw() #Redraw the figure
#                 dat.sub_min = 0
#                 dat.sub_max = (dat.nrow)

            elif key == 's':
                Button_functions.set_time(dat)


            elif key == '1' or key =="2"  or key == '3' or key == '4'or key == '5'or key == '6'or key == '7'or key == '8'or key == '9' or key == '0'or key == 'F1'or key == 'F1'or key == 'F2'or key == 'F3'or key == 'F4'or key == 'F5'or key == 'F6'or key == 'F7'or key == 'F8'or key == 'F9':
                if key[0] == "F":
                    key = "1" + key[1]
                dat.key = key
                Button_functions.annotate(self,dat)
                # idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick
                # dat.bexported = False #Reset the PCE export flag
                # if key == 1:
                #     # dat.df.iloc[np.arange((idx-4),(idx + 4),1),16] = 2
                #     dat.df.iloc[idx,dat.df.columns.get_loc("PCE")] = 1
                # else:
                #     dat.df.iloc[idx,dat.df.columns.get_loc("PCE")] = key

                # print(key)
##                print('mark PCE')
            # elif key == 'd':
            #     dat.dive_num = dat.dive_num + 1
            #     dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc("DIVE")] = dat.dive_num
            #     print('marked Dive '+ str(dat.dive_num))
            # elif key == "u":
            #     dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc("PCE")] = 9

            # elif key == 'q':
            #     dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc("BEHAV")] = 1
            #     print("marked descend")
            # elif key == 'w':
            #     dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc("BEHAV")] = 2
            #     print("marked forage")
            # elif key == 'e':
            #     dat.df.iloc[np.arange(dat.dive_min,dat.dive_max),dat.df.columns.get_loc("BEHAV")] = 3
            #     print("marked ascend")

            elif key == 'y':
                print("axes reset")
                i_int = round(dat.nrow/2000)
                #Specificy the rows to plot
                row_to_use = np.arange(0,(dat.nrow),i_int)
                # Plot each column individually
                for col in dat.p_cols:
                    # Select the data for the current column
                    x_data = dat.df.iloc[row_to_use, dat.time_col]
                    y_data = dat.df.iloc[row_to_use, col]

                    # Exclude NA values
                    non_na_mask = ~y_data.isna()
                    x_data = x_data[non_na_mask]
                    y_data = y_data[non_na_mask]

                    # Plot the current column
                    dat.ax_main.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
                # dat.ax_main.plot(dat.df.iloc[np.arange(0,(dat.nrow),1000),dat.time_col],dat.df.iloc[np.arange(0,(dat.nrow),1000),dat.p_cols])
                dat.figure_main.canvas.draw()

            elif key == "space":
                Keyboard_functions.vid_play(self, dat)


            elif key == "r":
                Keyboard_functions.vid_rewind(self, dat)

            elif (key == "Right") or (key == "n"):# and dat.frame < dat.frame_count:
                Keyboard_functions.frame_fwd(self,dat)

            elif (key == "Left") or (key == "b"):
                Keyboard_functions.frame_bkwd(self, dat)

            elif (key == "space" or key =="r") and dat.playing == True:
                dat.playing = False
##                dat.panel.configure(image = image)
##                dat.panel.image = dat.image
                self.update()
            elif key == 'x':
                sub_diff = dat.sub_max - dat.sub_min
                dat.sub_max = dat.sub_max + int(sub_diff/2)
                dat.sub_min = dat.sub_min + int(sub_diff/2)
                if dat.sub_max > (dat.nrow):
                    dat.sub_max = (dat.nrow)
                    dat.sub_min = dat.sub_max - 100
                dat.vline2.remove()
                dat.vline2 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_max,dat.time_col],color = 'red')
                dat.vline1.remove()
                dat.vline1 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_min,dat.time_col],color = 'red')
                dat.figure_main.canvas.draw()

                if dat.sub_min > dat.sub_max:
                    sub_min_hold = dat.sub_min
                    dat.sub_min = dat.sub_max
                    dat.sub_max = sub_min_hold
                if (dat.sub_max - dat.sub_min) > 2000:
                    dat.zoom_int = round((dat.sub_max - dat.sub_min)/2000)
                else:
                    dat.zoom_int = 1
                # if (dat.sub_max - dat.sub_min) > 1440: # 1440 is 1 min at 24Hz
                #     dat.zoom_int = dat.frequency
                # elif (dat.sub_max - dat.sub_min) > 14400:
                #     dat.zoom_int = 1000
                # else:
                #     dat.zoom_int = 1

                dat.ax_zoom.cla() #Clear axes
                # Extracting the data for zoom plot
                #Specificy the rows to plot
                row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
                # Plot each column individually
                for col in dat.p_cols:
                    # Select the data for the current column
                    x_data = dat.df.iloc[row_to_use, dat.time_col]
                    y_data = dat.df.iloc[row_to_use, col]

                    # Exclude NA values
                    non_na_mask = ~y_data.isna()
                    x_data = x_data[non_na_mask]
                    y_data = y_data[non_na_mask]

                    # Plot the current column
                    dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
                # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
                dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
##                dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
##                dat.ax_zoom.axvline(x = dat.df.iloc[dat.dive_max,dat.time_col],color = 'blue')
##                dat.ax_zoom.axvline(x = dat.df.iloc[dat.dive_min,dat.time_col],color = 'green')
                dat.figure_zoom.canvas.draw() #Redraw the figure
                self.update()

            elif key == 'z':
                sub_diff = dat.sub_max - dat.sub_min
                dat.sub_max = dat.sub_max - int(sub_diff/2)
                dat.sub_min = dat.sub_min - int(sub_diff/2)
                if dat.sub_max < 0:
                    dat.sub_max = 100
                    dat.sub_min = 0
                dat.vline2.remove()
                dat.vline2 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_max,dat.time_col],color = 'red')
                dat.vline1.remove()
                dat.vline1 = dat.ax_main.axvline(x = dat.df.iloc[dat.sub_min,dat.time_col],color = 'red')
                dat.figure_main.canvas.draw()

                if dat.sub_min > dat.sub_max:
                    sub_min_hold = dat.sub_min
                    dat.sub_min = dat.sub_max
                    dat.sub_max = sub_min_hold
                if (dat.sub_max - dat.sub_min) > 2000:
                    dat.zoom_int = round((dat.sub_max - dat.sub_min)/2000)
                else:
                    dat.zoom_int = 1
                # if (dat.sub_max - dat.sub_min) > 1440: # 1440 is 1 min at 24Hz
                #     dat.zoom_int = dat.frequency
                # elif (dat.sub_max - dat.sub_min) > 14400:
                #     dat.zoom_int = 1000
                # else:
                #     dat.zoom_int = 1

                dat.ax_zoom.cla() #Clear axes
                # Extracting the data for zoom plot
                #Specificy the rows to plot
                row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
                # Plot each column individually
                for col in dat.p_cols:
                    # Select the data for the current column
                    x_data = dat.df.iloc[row_to_use, dat.time_col]
                    y_data = dat.df.iloc[row_to_use, col]

                    # Exclude NA values
                    non_na_mask = ~y_data.isna()
                    x_data = x_data[non_na_mask]
                    y_data = y_data[non_na_mask]

                    # Plot the current column
                    dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
                # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
                dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
##                dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
##                dat.ax_zoom.axvline(x = dat.df.iloc[dat.dive_max,dat.time_col],color = 'blue')
##                dat.ax_zoom.axvline(x = dat.df.iloc[dat.dive_min,dat.time_col],color = 'green')
                dat.figure_zoom.canvas.draw() #Redraw the figure
                self.update()
            else:
                print("not a valid option")

    def vid_play(self, dat):
        if dat.playing == False:
            dat.playing = True
            frame_count = dat.frame_count - 1
            while (dat.frame < frame_count and dat.playing == True):
    ##                    print(dat.frame)
                dat.frame = dat.frame + 1
                # dat.vid.set(1,dat.frame)
                Keyboard_functions.plot_frame(self,dat)#dat.plot_frame() #Plot the selected image
                self.update()

                dat.sub_max = dat.sub_max + 1
                dat.sub_min = dat.sub_min + 1
            dat.tbar.set(dat.frame)
            dat.playing = False
        else:
            dat.playing = False
            self.update()
    def vid_rewind(self,dat):
        if dat.playing == False:
            dat.playing = True
            # frame_count = dat.frame_count - 1
            while (dat.frame >= 0 and dat.playing == True):
    ##                    print(dat.frame)
                dat.frame = dat.frame - 1
                dat.vid.set(1,dat.frame)
                Keyboard_functions.plot_frame(self,dat)#dat.plot_frame() #Plot the selected image
                self.update()

                dat.sub_max = dat.sub_max - 1
                dat.sub_min = dat.sub_min - 1

            dat.playing = False
        else:
            dat.playing = False
            self.update
    def frame_fwd(self,dat):
        if dat.frame < dat.frame_count:
            dat.frame = dat.frame + 1
            dat.vid.set(1,dat.frame)
            Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()
            # idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick
    ##                dat.sub_min = idx - 500
    ##                dat.sub_max = idx + 500
            ipick = dat.frame_date

    ##                if dat.sub_min > dat.sub_max:
    ##                    sub_min_hold = dat.sub_min
    ##                    dat.sub_min = dat.sub_max
    ##                    dat.sub_max = sub_min_hold
    ##                if (dat.sub_max - dat.sub_min) > 1440: # 1440 is 1 min at 24Hz
    ##                    dat.zoom_int = 24
    ##                elif (dat.sub_max - dat.sub_min) > 14400:
    ##                    dat.zoom_int = 1000
    ##                else:
    ##                    dat.zoom_int = 1
            #Update the zoom plot

            #Move the plotted data window forwards
            if dat.sub_max < len(dat.df):
                dat.sub_max = dat.sub_max + 1
                dat.sub_min = dat.sub_min + 1
            dat.ax_zoom.cla() #Clear axes
            # Extracting the data for zoom plot
            #Specificy the rows to plot
            row_to_use = np.arange(dat.sub_min,dat.sub_max,dat.zoom_int)
            # Plot each column individually
            for col in dat.p_cols:
                # Select the data for the current column
                x_data = dat.df.iloc[row_to_use, dat.time_col]
                y_data = dat.df.iloc[row_to_use, col]

                # Exclude NA values
                non_na_mask = ~y_data.isna()
                x_data = x_data[non_na_mask]
                y_data = y_data[non_na_mask]

                # Plot the current column
                dat.ax_zoom.plot(x_data, y_data, label=dat.df.iloc[:,col].name)
            # dat.ax_zoom.plot(dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.time_col],dat.df.iloc[np.arange(dat.sub_min,dat.sub_max,dat.zoom_int),dat.p_cols],label = dat.df.iloc[:,dat.p_cols].columns) #Plot new values
            dat.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            dat.vline3 = dat.ax_zoom.axvline(x = ipick,color = 'green')
            dat.figure_zoom.canvas.draw() #Redraw the figure

            #Update Audio plot
            if dat.audio_present == True:
                dat.sub_window = dat.sub_max - dat.sub_min
                dat.aud_window = int((dat.sub_window*640)/2)
                dat.aud_point = int(dat.frame*640)
                if dat.frame > 1:
                    dat.aud_min = dat.aud_point - dat.aud_window
                else:
                    dat.aud_min = dat.aud_point
                dat.aud_max = dat.aud_point + dat.aud_window

                dat.ax_audio.cla()
                dat.ax_audio.plot(dat.audio.iloc[np.arange(dat.aud_min,dat.aud_max,10),1],dat.audio.iloc[np.arange(dat.aud_min,dat.aud_max,10),0])
                dat.vline_aud = dat.ax_audio.axvline(x = dat.audio.iloc[dat.aud_point,1],color = 'red')
                dat.figure_audio.canvas.draw()

            self.update()

            #Update the audio plotted
    def frame_bkwd(self,dat):
        if dat.frame > 0:
            #Move the plotted data window backwards
            if dat.sub_min > 2:
                dat.sub_min = dat.sub_min - 1
                dat.sub_max = dat.sub_max - 1
            dat.frame = dat.frame - 1
            dat.vid.set(1,dat.frame)
            Keyboard_functions.plot_frame(self,dat)#dat.plot_frame()

            #Update Audio plot
            if dat.audio_present == True:
                dat.sub_window = dat.sub_max - dat.sub_min
                dat.aud_window = int((dat.sub_window*640)/2)
                dat.aud_point = int(dat.frame*640)
                if dat.frame > 1:
                    dat.aud_min = dat.aud_point - dat.aud_window
                else:
                    dat.aud_min = dat.aud_point
                dat.aud_max = dat.aud_point + dat.aud_window

            self.update()
    def jump_frame(self,dat):
        dat.frame = dat.frame + (dat.IMU_date - dat.frame_date).total_seconds()*dat.fps

    def plot_frame(self,dat):

        # global img_th,img_blur,img_edge,img_bright

        dat.bconfig = False #Reset the config flag

        s, image = dat.vid.read()


        time = float(dat.frame)/dat.fps
        # time = self.vid.get(cv2.CAP_PROP_POS_MSEC)/1000
        # print(f"Frame {self.frame}")
        # print(f"Frame time internal {time}")
        # print(f"Frame time manual {float(self.frame)/self.fps}")
        tnow = datetime.utcnow()
        dat.playback_speed = dat.speed_slider.get()
        dat.frame_date = dat.vid_start_date + timedelta(seconds = time)

        # s, image = self.vid.read()
        #Copythe image to allow YOLO model prediction
        yolo_image = image.copy()
        # dat.tbar.set(dat.frame)

        blur1 = dat.blur1.get()
        if(blur1 % 2 == 0):
            blur1 = blur1 + 1
        blur2 = dat.blur2.get()
        if(blur2 % 2 == 0):
            blur2 = blur2 + 1

        canny1 = dat.canny1.get()
        if(canny1 % 2 == 0):
            canny1 = canny1 + 1
        canny2 = dat.canny2.get()
        if(canny2 % 2 == 0):
            canny2 = canny2 + 1

        if(dat.img_edge.get() ==1):
##            print("canny: "+str(self.canny1.get()))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)

            edges = cv2.Canny(blur,dat.canny1.get(),dat.canny2.get(),L2gradient = True)
            edges = cv2.dilate(edges,np.ones((2,2), np.uint8),iterations = 2)
            edges = cv2.erode(edges,np.ones((2,2), np.uint8),iterations = 2)

            di = cv2.dilate(edges,np.ones((3,3), np.uint8),iterations = 2)

            edges2 = cv2.Canny(di,dat.canny1.get(),dat.canny2.get(),L2gradient = True)

            image = edges2

        elif(dat.img_th.get() == 1):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
        ##    blur = cv2.bilateralFilter(er,15,int_range/5,int_range/5)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)
##            ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #transform image to binary image
            ret3,th3 =  cv2.threshold(blur,canny1,canny2,cv2.THRESH_BINARY_INV)
##            th3 = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, canny1,canny2)
            image = th3
        elif(dat.img_blur.get() ==1):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
        ##    blur = cv2.bilateralFilter(er,15,int_range/5,int_range/5)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)
            image = blur
        elif(dat.img_bright.get() == 1):
            yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Equalize the histogram of the Y channel (luma)
            y, u, v = cv2.split(yuv_image)
            y = cv2.equalizeHist(y)

            # Merge the equalized Y channel back with the U and V channels
            yuv_image = cv2.merge((y, u, v))
            # Convert the YUV image back to the BGR color space
            bright_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            image = bright_image
        elif(dat.horison_detect.get()):
            font = cv2.FONT_HERSHEY_SIMPLEX
            canny1 = 5
            canny2 = canny1/4

            blur1 = 13

            bilat_blur = 10

            num_down = 2       # number of downsampling steps
            num_bilateral = bilat_blur  # number of bilateral filtering steps

            img_rgb = image

            # downsample image using Gaussian pyramid
            img_color = img_rgb
            for _ in range(num_down):
                img_color = cv2.pyrDown(img_color)

            # repeatedly apply small bilateral filter instead of
            # applying one large filter
            for _ in range(num_bilateral):
                img_color = cv2.bilateralFilter(img_color, d=9,
                                                sigmaColor=9,
                                                sigmaSpace=7)

            # upsample image to original size
            for _ in range(num_down):
                img_color = cv2.pyrUp(img_color)


            #------------Contrast-----------#
            lab= cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1,1))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            #Crop the image to exclude the time stamp
            crop = final[0:525, 100:1280]

            #Crop image (excluding timestamp) to see if horison is present
            crop2 = image[100:455, 100:1180]
            int_range = int(crop2.max())-int(crop2.min())

            #Only continue if the range of pixel intensities are above a threshold
            if int_range > 0:
            ##    int_range = np.int(crop.max())-np.int(crop.min())
                img_use = crop.copy()
        ##        img_use = (125-img_use)
            ##    if (test == 1):cv2.imwrite("img_" + sf.zfill(7) +  ".jpg", img_use)
                gray = cv2.cvtColor(img_use,cv2.COLOR_BGR2GRAY)
                img_use = gray
                kernel1 = np.ones((1,1), np.uint8)
                er = cv2.erode(gray, kernel1, iterations = 20)
            ##    blur = cv2.bilateralFilter(er,15,int_range/5,int_range/5)
                blur = cv2.GaussianBlur(er,(blur1,blur1),0)
                edges = cv2.Canny(blur,canny1,canny2)
                edges = cv2.dilate(edges,np.ones((2,2), np.uint8),iterations = 2)
                edges = cv2.erode(edges,np.ones((2,2), np.uint8),iterations = 2)

                di = cv2.dilate(edges,np.ones((3,3), np.uint8),iterations = 2)

                edges2 = cv2.Canny(di,6,1,L2gradient = True)
                cnts, _= cv2.findContours(edges2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                c_wdt = 0
                c_hold = None
                box_hold = None
                dr_hold = None
                for c in cnts:
                    dR = 0
                    #compute the bounding box for the contour, draw and update text
                    (x,y,w,h) = cv2.boundingRect(c)
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    dA =  math.sqrt( (box[0][0]- box[1][0] )**2 + ( box[0][1]-box[1][1] )**2 )
                    dB =  math.sqrt( (box[1][0]- box[2][0] )**2 + ( box[1][1]-box[2][1] )**2 )
                    if dA == 0 or dB == 0:
                        continue
                    if dA < dB:
                        dR = dA/dB
                    else:
                        dR = dB/dA
                    if dR <= 0.05:
                        if w > c_wdt:
                            c_wdt = w
                            c_hold = c.copy()
                            box_hold = box.copy()
                            dR_hold = dR

                if c_hold is None:
                    angle_contour = "NA"
                    cv2.putText(crop, str(time) + ": " + str(angle_contour),(10,50), font, 1,(0,0,255),2,cv2.LINE_AA)
                    c_hold = None
                    box_hold = None
                    dr_hold = None
                else:
                    cv2.drawContours(crop,c_hold,-1,(125,125,0),2)
                    cv2.drawContours(crop,[box_hold],0,(0,0,255),2)
                    rows,cols = img_use.shape[:2]
                    [vx,vy,x,y] = cv2.fitLine(c_hold, cv2.DIST_L2,0,0.01,0.01)
                    lefty = int((-x*vy/vx) + y)
                    righty = int(((cols-x)*vy/vx)+y)
                    cv2.line(crop,(cols-1,righty),(0,lefty),(125,125,0),2)
                    angle_contour = round(np.arctan2(righty-lefty,(cols-1)-0)*180/np.pi,0)
                    cv2.putText(crop,str(time) + ": " +  str(angle_contour),(10,50), font, 1,(0,0,255),2,cv2.LINE_AA)
                image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                idx = (np.abs(dat.df.iloc[:,dat.time_col] - dat.frame_date)).argmin() #Find the row in the df with the nearest value to ipick
                dat.df.loc[idx,"horison_angle"] = angle_contour
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Show the YOLO results if it is loaded
        if(dat.yolo_loaded ==1):
            #Predict on the frame using the loaded model
            yolo_predict = dat.yolo_model.predict(yolo_image,
            conf = 0.3,
            iou = 0.6,
            nms = True,
            max_det = 50,
            seed = 42,
            verbose=False)
            #Read the results
            model_result = yolo_predict[0]

            #Classifier
            if model_result.probs is not None:
                #Extract the results and label
                cls =  round(model_result.probs.top1,2)
                if cls == 0:
                    cls = 1
                    cls_name = "Krill"
                    font_color = (255, 12, 12) #Red color text
                else:
                    cls=0
                    cls_name = "No_Krill"
                    font_color = (36, 255, 12) #Green color text
                #probability values
                confs =  round(model_result.probs.top1conf.item(),2)
                #Text to be printed on image
                text = f'Class: {cls_name}, Conf: {confs}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.9
                font_thickness = 2

                # Get text size
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

                # Calculate the starting point for the text (bottom-right corner)
                image_height, image_width = image.shape[:2]
                text_position = (image_width - text_size[0] - 10, image_height - 10)

                # Draw the text on the image
                cv2.putText(image, text, text_position, font, font_scale, font_color, font_thickness)

            #Object detector
            elif model_result.boxes is not None:
                #Find all the bounding boxes
                bboxes = model_result.boxes.xyxy.int().tolist()
                #Extract the model results
                cls =  model_result.boxes.cls.int().tolist()
                #Extract the probabilities
                confs =  model_result.boxes.conf.tolist()
                #Mergee the boxes, labels and probs
                bb_cls_cf = [(bb,cl,cf) for cl,bb,cf in zip(cls,bboxes,confs)]
                #Draw the boxes on the image
                for  bb,cl,cf in bb_cls_cf:
                    b_x1,b_y1,b_x2,b_y2 =  bb
                    cv2.rectangle(image,(b_x1,b_y1),(b_x2,b_y2), (0, 255, 0), 2)
                    cv2.putText(image, str(cf), (b_x1, b_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    # box_count+=1
            else:
                pass

        # image = cv2.resize(image,(self.video_width,int(self.video_width/1.5)))
        image = cv2.resize(image,(dat.video_frame.winfo_width(),int(dat.video_frame.winfo_width()/1.5)))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        #wait to make sure the selected playback speed is adhered # TODO:

        while(((datetime.utcnow()-tnow).total_seconds()*1000 < (1000/dat.playback_speed)) and ((datetime.utcnow()-tnow).total_seconds()*1000 < (1000))):
            pass

        dat.panel.configure(image = image)
        dat.panel.image = image

# class Menu_functions_FILTER(Data_Frame):
#     def apply_filter(self,dat,filter):
#
#         # def choose_cols(self,dat):
#         w2 = tk.Toplevel(self)
#         w2.title("Choose column")
#         w2.geometry("1000x1000")
#         def fapply(self,dat):
#             c = 0
#             # dat.p_fcols = []
#             for i in dat.col_names:
#                 x = eval("dat.cf" + str(c))
#                 if(x.get() == 1):
#                     dat.p_fcols = c
#                     # print(dat.p_fcols)
#                 c = c+1
#     ##            self.plot_dat(self,dat)
#         #Apply lulu filter
#         # 0 - LULU up
#         # 1 - LULU down
#         # 2 - gradient
#         # 3 - rolling mean
#         # 4 - Notch filter
#             if filter == 0:
#                 lulu1 = dat.lulu1.get()
#                 lulu2 = dat.lulu2.get()
#                 dat.df.cfilter = lulu_filter().bpu(dat.df.iloc[:,dat.p_fcols],lulu1,lulu2)
#             elif filter == 1:
#                 lulu1 = dat.lulu1.get()
#                 lulu2 = dat.lulu2.get()
#                 dat.df.cfilter = lulu_filter().bpl(dat.df.iloc[:,dat.p_fcols],lulu1,lulu2)
#             elif filter == 2:
#                 dat.df.cfilter = np.gradient(dat.df.iloc[:,dat.p_fcols])
#             elif filter == 3:
#                 smooth = dat.slider2.get()
#                 dat.df.cfilter = dat.df.iloc[:,dat.p_fcols].rolling(smooth,center = True).mean()
#             # elif filter == 4:
#             #     f = 25
#             #     a = 2.4
#             #     q = 10
#             #     b, a = signal.iirnotch(a, q, f)
#             #     dat.df.cfilter = signal.filtfilt(b, a, dat.df.iloc[:,dat.p_fcols])
#
#             w2.destroy()
#
#         btn_close = tk.Button(w2,text = "Apply",command = lambda:fapply(self,dat))
#         btn_close.pack()
#         c = 0
#         iy = 50
#         ix = 0
#         for i in dat.col_names:
#     ##            exec("dat.c" + str(c) +" = tk.IntVar()")
#             # tk.Checkbutton(w2, text=i,variable=eval("dat.c" + str(c)), onvalue=1, offvalue=0 ).pack()
#             eval("dat.cf" + str(c) + ".set(0)")
#             tk.Radiobutton(w2, text=i,padx = 20, variable=eval("dat.cf" + str(c)), value=1).place(x = ix, y=iy)
#             c = c +1
#             iy = iy + 100
#             if iy > 1000:
#                 iy = 50
#                 ix = ix + 150
#         # print(f"Applying lulu with v1: {lulu1} and v2: {lulu2}")
#
