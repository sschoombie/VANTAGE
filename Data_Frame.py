
import tkinter as tk

import cv2
from PIL import Image
from PIL import ImageTk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
from datetime import datetime,timedelta
import numpy as np
import os
from pytz import utc

# Window to print debugging messages to
class DebugWindow():
    def __init__(self, textbox):
        self.textbox = textbox
    def write(self,text):
        self.textbox.insert(tk.END,text)
        self.textbox.see(tk.END)
    def flush(self):
        pass
    #DebugWindow end

#This is a temporary data frame used when exporting the config file
class temp_Data_Frame:
    def __init__(self):
        pass

class Data_Frame:
    def __init__(self,sub_min = 0, sub_max = 1000):
        #Input data information
        self.filename = "" #Path to INPUT file
        self.out_filename = ""   #Path to OUTPUT file
        self.col_names = ["None"]#Column names of the dataset
        self.p_cols = [2,3,4,5] #Which columns should be plotted - Default is the order of the AxyTrek tags
        self.time_col_string = "Timestamp"
        self.IMU_date = 0        #Flag to show if the date has been set
        self.smooth = 1          #Should the data be smoothed? 1- Yes, 0 - No
        self.frequency = 24      #Sampling frequency of the data
        #Basic data navigation
        self.sub_min = sub_min  #Pointer to the FIRST datapoint to be plotted
        self.sub_max = sub_max  #Pointer to the LAST datapoint to be plotted
        self.zoom_ipick = 0     #Holder for a point selected by the user
        self.zoom_int = 1       #Interval for plots (i.e. interval between points - this will increase with more data to assist plotting)
        self.current_label_col = "PCE"
        self.all_label_col = ["PCE"]
        #Audio
        self.aud_min = 0
        self.aud_max = 16000

        #Dive analysis
        self.dive_max_ipick = 0
        self.dive_min_ipick = 0
        self.dive_max = 0
        self.dive_min = 0

        #Flags
        self.audio_present = False         #Flag to see if audio is loaded
        self.video_loaded = False          #Flag to see if video has been loaded
        self.df_loaded = False             #Flag to see if data has been loaded
        self.dives_loaded = False          #Flag to see if dives have been loaded
        self.out_file_loaded = False       #Flag to see if OUT file have been loaded
        self.bwd_set = False               #video directory set?
        self.bplot = False                 #Data plotted?
        self.bexported = True              #Has the OUT file been exported?
        self.bconfig = False               #Has the config file been saved?
        self.view_only = False             #if True - no changes are allowed to the annotations
        self.bdives = False                #Flag to show that dives are being calculated
        self.date_set = 0                  #Flag to see if the video and data times have been synched at any point
        self.vid_date_set= 0               #Flag to see if the video and data times have been synched for the CURRENT video clip
        self.playing = False               #Flag to see if the video is currently playing
        self.temp_done = True              #This is used to hold pause processes while waiting for others to finish
        self.yolo_loaded = 0               #Check if a YOLO model is loaded or not
        self.squash_loaded = False         #Check if the squashed image is loaded

        #Video analysis
        self.frame = -1                   #Keeping track of the current video frame
        self.video_offset = timedelta(0)  #Offset in time between the video and accompanying data
        # self.brightness = 0
        self.wd = None             #path to the location of the video files
        self.v_num = 0                    #Video number (i.e. file number in the working directory)
        self.vt_present = False           #Is the vt (video times) file present in the working directory
        self.video_width = 932            #Plotting resolution (width in pixels)
        self.playback_speed = 24          #adjustable playback speed - in frames per second (adjusted with slider)
        self.bright = 0                   #Brighten image - 0 = no; 1 = yes
        self.bsquash = True               #Is the squased image present?
        self.key = 0                      #Holder for the key being pressed on the keyboard

        #Variables used in interactive widgets (e.g. sliders, or checkboxes)
        self.annotate_selection = tk.BooleanVar()      #Checkbox menu to decide if a single point should be annotated
        self.norm = tk.IntVar()                     #Checkbox for normalization of data
        self.blur1 = tk.IntVar()#.set(13)           #Slider1 for blurring
        self.blur2 = tk.IntVar()#.set(13)           #Slider2 for blurring
        self.canny1 = tk.IntVar()#.set(6)           #Slider1 for canny edge detection threshold
        self.canny2 = tk.IntVar()#.set(1)           #Slider2 for canny edge detection threshold
        self.lulu1 = tk.IntVar()#.set(1)            #Slider1 for LULU filter
        self.lulu2 = tk.IntVar()#.set(1)            #Slider2 for LULU filter
        self.img_blur = tk.IntVar()                 #Checkbox for image blurring
        self.img_th = tk.IntVar()                   #Checkbox for image thresholding
        self.img_edge = tk.IntVar()                 #Checkbox for edge detection
        self.img_bright = tk.IntVar()               #Checkbox for brightning of image
        self.horison_detect = tk.BooleanVar()       #Menu checkbox to see if the horizon should be estimated



    def plot_frame(self):

        # global img_th,img_blur,img_edge,img_bright

        self.bconfig = False #Reset the config flag

        s, image = self.vid.read()


        time = float(self.frame)/self.fps
        # time = self.vid.get(cv2.CAP_PROP_POS_MSEC)/1000
        # print(f"Frame {self.frame}")
        # print(f"Frame time internal {time}")
        # print(f"Frame time manual {float(self.frame)/self.fps}")
        tnow = datetime.utcnow()
        self.playback_speed = self.speed_slider.get()
        self.frame_date = self.vid_start_date + timedelta(seconds = time)

        # s, image = self.vid.read()
        #Copythe image to allow YOLO model prediction
        yolo_image = image.copy()
        self.tbar.set(self.frame)

        blur1 = self.blur1.get()
        if(blur1 % 2 == 0):
            blur1 = blur1 + 1
        blur2 = self.blur2.get()
        if(blur2 % 2 == 0):
            blur2 = blur2 + 1

        canny1 = self.canny1.get()
        if(canny1 % 2 == 0):
            canny1 = canny1 + 1
        canny2 = self.canny2.get()
        if(canny2 % 2 == 0):
            canny2 = canny2 + 1

        if(self.img_edge.get() ==1):
##            print("canny: "+str(self.canny1.get()))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)

            edges = cv2.Canny(blur,self.canny1.get(),self.canny2.get(),L2gradient = True)
            edges = cv2.dilate(edges,np.ones((2,2), np.uint8),iterations = 2)
            edges = cv2.erode(edges,np.ones((2,2), np.uint8),iterations = 2)

            di = cv2.dilate(edges,np.ones((3,3), np.uint8),iterations = 2)

            edges2 = cv2.Canny(di,self.canny1.get(),self.canny2.get(),L2gradient = True)

            image = edges2

        elif(self.img_th.get() == 1):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
        ##    blur = cv2.bilateralFilter(er,15,int_range/5,int_range/5)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)
##            ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  #transform image to binary image
            ret3,th3 =  cv2.threshold(blur,canny1,canny2,cv2.THRESH_BINARY_INV)
##            th3 = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, canny1,canny2)
            image = th3
        elif(self.img_blur.get() ==1):
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kernel1 = np.ones((1,1), np.uint8)
            er = cv2.erode(image, kernel1, iterations = 20)
        ##    blur = cv2.bilateralFilter(er,15,int_range/5,int_range/5)
            blur = cv2.GaussianBlur(er,(blur1,blur2),0)
            image = blur
        elif(self.img_bright.get() == 1):
            yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Equalize the histogram of the Y channel (luma)
            y, u, v = cv2.split(yuv_image)
            y = cv2.equalizeHist(y)

            # Merge the equalized Y channel back with the U and V channels
            yuv_image = cv2.merge((y, u, v))
            # Convert the YUV image back to the BGR color space
            bright_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
            image = bright_image


        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Show the YOLO results if it is loaded
        if(self.yolo_loaded ==1):
            #Predict on the frame using the loaded model
            yolo_predict = self.yolo_model.predict(yolo_image,
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
        image = cv2.resize(image,(self.video_frame.winfo_width(),int(self.video_frame.winfo_width()/1.5)))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        #wait to make sure the selected playback speed is adhered # TODO:

        while(((datetime.utcnow()-tnow).total_seconds()*1000 < (1000/self.playback_speed)) and ((datetime.utcnow()-tnow).total_seconds()*1000 < (1000))):
            pass

        self.panel.configure(image = image)
        self.panel.image = image

    def main_zoom(self,event):
        array = self.df.iloc[:,self.time_col]                  #Define the numeric date column as an array
##                ipick = (event.xdata)                  #when mouse is clicked, x-coordinate is saved
        ipick = mdates.num2date(event.xdata,utc)
        idx =  (np.abs(array - ipick)).argmin() #Find the row in the df with the nearest value to ipick
        if event.button == 3: #RIGHT
            self.sub_max = idx
            try:
                self.vline2.remove()
            except:
                pass
            self.vline2 = self.ax_main.axvline(x = ipick,color = 'red')
        elif event.button == 1: #LEFT
            self.sub_min = idx
            try:
                self.vline1.remove()
            except:
                pass
            self.vline1 = self.ax_main.axvline(x = ipick,color = 'red')
        elif event.button == 2:
            self.IMU_date = array[idx]
            # print(self.IMU_date)
            try:
                self.vline3.remove()
            except:
                pass
            self.vline3 = self.ax_zoom.axvline(x = ipick,color = 'green')

        self.figure_main.canvas.draw()   #Redraw the figure

          #Make sure the ranges stays within bounds of the dataframe
        if self.sub_min > self.sub_max:
            if self.sub_max == 0:
                self.sub_max = self.sub_min
            else:
                sub_min_hold = self.sub_min
                self.sub_min = self.sub_max
                self.sub_max = sub_min_hold
        # make sure there are no more than 2000 data points plotted
        if (self.sub_max - self.sub_min) > 2000:
            self.zoom_int = round((self.sub_max - self.sub_min)/2000)
        else:
            self.zoom_int = 1

        # if (self.sub_max - self.sub_min) > 1440: # 1440 is 1 min at 24Hz
        #     self.zoom_int = 24
        # elif (self.sub_max - self.sub_min) > 14400:
        #     self.zoom_int = 1000
        # else:
        #     self.zoom_int = 1

        self.ax_zoom.cla() #Clear axes
        row_to_use = np.arange(self.sub_min,self.sub_max,self.zoom_int)
        # Plot each column individually
        for col in self.p_cols:
            # Select the data for the current column
            x_data = self.df.iloc[row_to_use, self.time_col]
            y_data = self.df.iloc[row_to_use, col]

            # Exclude NA values
            non_na_mask = ~y_data.isna()
            x_data = x_data[non_na_mask]
            y_data = y_data[non_na_mask]

            # Plot the current column
            self.ax_zoom.plot(x_data, y_data, label=self.df.iloc[:,col].name)

        # self.ax_zoom.plot(self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.time_col],self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.p_cols],label = self.df.iloc[:,self.p_cols].columns) #Plot new values
        self.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
        self.vline3 = self.ax_zoom.axvline(x = ipick,color = 'green')
        self.figure_zoom.canvas.draw() #Redraw the figure

    def zoom_zoom(self,event):
        array = self.df.iloc[:,self.time_col]                  #Define the numeric date column as an array
##                ipick = (event.xdata)                  #when mouse is clicked, x-coordinate is saved
        ipick = mdates.num2date(event.xdata,utc)
        idx = (np.abs(array - ipick)).argmin() #Find the row in the df with the nearest value to ipick

        #Print the frame data to the debug window
        print(f"{self.df.iloc[:,self.time_col].name}: {self.df.iloc[idx,self.time_col]}")
        for col in self.p_cols:
            print(f"{self.df.iloc[:,col].name}: {self.df.iloc[idx,col]}")

        if event.button == 3: #RIGHT
            array = self.df.iloc[:,self.time_col]                  #Define the numeric date column as an array
##                ipick = (event.xdata)                  #when mouse is clicked, x-coordinate is saved
            ipick = mdates.num2date(event.xdata,utc)
            idx = (np.abs(array - ipick)).argmin() #Find the row in the df with the nearest value to ipick
            dive_max_hold = self.dive_max
            dive_max_ipick_hold = self.dive_max_ipick
            self.dive_max = idx
            self.dive_max_ipick = ipick
            if self.dive_min_ipick == 0:
                self.dive_min_ipick = self.dive_max_ipick
                self.dive_min = self.dive_max

            elif self.dive_min > self.dive_max:
                self.dive_min = self.dive_max
                self.dive_min_ipick = self.dive_max_ipick
                self.dive_max = dive_max_hold
                self.dive_max_ipick = dive_max_ipick_hold


        elif event.button == 1: #LEFT
            array = self.df.iloc[:,self.time_col]                  #Define the numeric date column as an array
##                ipick = (event.xdata)                  #when mouse is clicked, x-coordinate is saved
            ipick = mdates.num2date(event.xdata,utc)
            idx = (np.abs(array - ipick)).argmin() #Find the row in the df with the nearest value to ipick
            dive_min_hold = self.dive_min
            dive_min_ipick_hold = self.dive_min_ipick
            self.dive_min = idx
            self.dive_min_ipick = ipick

            if self.dive_max_ipick == 0:
                self.dive_max_ipick = self.dive_min_ipick
                self.dive_max = self.dive_min

            elif self.dive_min > self.dive_max:
                self.dive_max = self.dive_min
                self.dive_max_ipick = self.dive_min_ipick
                self.dive_min = dive_min_hold
                self.dive_min_ipick = dive_min_ipick_hold


        elif event.button == 2:
            self.IMU_date = array[idx]
            self.sub_min = idx -50
            self.sub_max = idx + 50
            self.zoom_ipick = ipick
##            frame_num = int(frame_count*id_perc)
##            image = plotframe(frame_num)
##            panel.configure(image = image)
##            panel.image = image
            self.vline3.remove()
            self.vline3 = self.ax_zoom.axvline(x = ipick,color = 'green')
            self.zoom_int = 1

        self.figure_main.canvas.draw()   #Redraw the figure


        # make sure there are no more than 2000 data points plotted
        # if (self.sub_max - self.sub_min) > 2000:
        #     self.zoom_int = round((self.sub_max - self.sub_min)/2000)
        # else:
        #     self.zoom_int = 1


        # if self.sub_min > self.sub_max:
        #     sub_min_hold = self.sub_min
        #     self.sub_min = self.sub_max
        #     self.sub_max = sub_min_hold
        # if (self.sub_max - self.sub_min) > 1440: # 1440 is 1 min at 24Hz
        #     self.zoom_int = 24
        # elif (self.sub_max - self.sub_min) > 14400:
        #     self.zoom_int = 1000
        # else:
        #     self.zoom_int = 1

        self.ax_zoom.cla() #Clear axes
        row_to_use = np.arange(self.sub_min,self.sub_max,self.zoom_int)
        # Plot each column individually
        for col in self.p_cols:
            # Select the data for the current column
            x_data = self.df.iloc[row_to_use, self.time_col]
            y_data = self.df.iloc[row_to_use, col]

            # Exclude NA values
            non_na_mask = ~y_data.isna()
            x_data = x_data[non_na_mask]
            y_data = y_data[non_na_mask]

            # Plot the current column
            self.ax_zoom.plot(x_data, y_data, label=self.df.iloc[:,col].name)

        # self.ax_zoom.plot(self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.time_col],self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.p_cols],label = self.df.iloc[:,self.p_cols].columns) #Plot new values
        self.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
        # if(self.zoom_ipick < self.sub_max and self.zoom_ipick > self.sub_min):
        #     self.vline3 = self.ax_zoom.axvline(x = self.zoom_ipick,color = 'green')
        self.vline3 = self.ax_zoom.axvline(x = self.zoom_ipick,color = 'green')
        if(self.dive_max < self.sub_max and self.dive_max > self.sub_min):
            self.vline4 = self.ax_zoom.axvline(x = self.dive_max_ipick,color = 'blue')
        if(self.dive_min < self.sub_max and self.dive_min > self.sub_min):
            self.vline5 = self.ax_zoom.axvline(x = self.dive_min_ipick,color = 'red')
##        self.ax_zoom.axvline(x = self.df.iloc[self.dive_max,self.time_col],color = 'blue')
##        self.ax_zoom.axvline(x = self.df.iloc[self.dive_min,self.time_col],color = 'green')
        self.figure_zoom.canvas.draw() #Redraw the figure

        #Update Audio plot
        if self.audio_present == True:
            self.sub_window = self.sub_max - self.sub_min
            self.aud_window = int((self.sub_window*640)/2)
            self.aud_point = int(self.frame*640)
            if self.frame > 1:
                self.aud_min = self.aud_point - self.aud_window
            else:
                self.aud_min = self.aud_point
            self.aud_max = self.aud_point + self.aud_window
            self.ax_audio.cla()
            self.ax_audio.plot(self.audio.iloc[np.arange(self.aud_min,self.aud_max,10),1],self.audio.iloc[np.arange(self.aud_min,self.aud_max,10),0])
            # self.vline_aud = self.ax_audio.axvline(x = self.dive_min_ipick,color = 'red')
            self.figure_audio.canvas.draw()
        # print(f"sub_min: {self.sub_min}")
        # print(f"sub_max: {self.sub_max}")
        # print(f"dive_min: {self.dive_min}")
        # print(f"dive_max: {self.dive_max}")
        # print(f"dive_min_ipick: {self.dive_min_ipick}")
        # print(f"dive_max_ipick: {self.dive_max_ipick}")



    def zoom_scroll(self,event):
        if event.button == 'up': #When scrolling up
##            print("up")
            self.sub_min = self.sub_min + 100
            self.sub_max = self.sub_max - 100
            if (self.sub_max - self.sub_min) < 100:
                self.sub_min = self.sub_min -100
                self.sub_max = self.sub_max + 100
            if (self.sub_max - self.sub_min) > 2000:
                self.zoom_int = round((self.sub_max - self.sub_min)/2000)
            else:
                self.zoom_int = 1
            #Redraw the main and zoom figures with the updated ranges
            self.ax_zoom.cla()
            row_to_use = np.arange(self.sub_min,self.sub_max,self.zoom_int)
            # Plot each column individually
            for col in self.p_cols:
                # Select the data for the current column
                x_data = self.df.iloc[row_to_use, self.time_col]
                y_data = self.df.iloc[row_to_use, col]

                # Exclude NA values
                non_na_mask = ~y_data.isna()
                x_data = x_data[non_na_mask]
                y_data = y_data[non_na_mask]

                # Plot the current column
                self.ax_zoom.plot(x_data, y_data, label=self.df.iloc[:,col].name)

            # self.ax_zoom.plot(self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.time_col],self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.p_cols],label = self.df.iloc[:,self.p_cols].columns)
            self.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            if(self.dive_max < self.sub_max and self.dive_max > self.sub_min):
                self.vline4 = self.ax_zoom.axvline(x = self.dive_max_ipick,color = 'blue')
            if(self.dive_min < self.sub_max and self.dive_min > self.sub_min):
                self.vline5 = self.ax_zoom.axvline(x = self.dive_min_ipick,color = 'red')
            self.figure_zoom.canvas.draw()

            self.vline1.remove()
            self.vline1 = self.ax_main.axvline(x = self.df.iloc[self.sub_min,self.time_col],color = 'red')
            self.figure_main.canvas.draw()
            self.vline2.remove()
            self.vline2 = self.ax_main.axvline(x = self.df.iloc[self.sub_max,self.time_col],color = 'red')
            self.figure_main.canvas.draw()

        elif event.button == 'down': #When scrolling down
            self.sub_min = self.sub_min - 100
            self.sub_max = self.sub_max + 100
            if self.sub_min < 0:
                self.sub_min = 0
            if self.sub_max > (self.nrow):
                self.sub_max = (self.nrow)
            if (self.sub_max - self.sub_min) > 2000:
                self.zoom_int = round((self.sub_max - self.sub_min)/2000)
            else:
                self.zoom_int = 1
            #Redraw the main and zoom figures with the updated ranges
            self.ax_zoom.cla()
            row_to_use = np.arange(self.sub_min,self.sub_max,self.zoom_int)
            # Plot each column individually
            for col in self.p_cols:
                # Select the data for the current column
                x_data = self.df.iloc[row_to_use, self.time_col]
                y_data = self.df.iloc[row_to_use, col]

                # Exclude NA values
                non_na_mask = ~y_data.isna()
                x_data = x_data[non_na_mask]
                y_data = y_data[non_na_mask]

                # Plot the current column
                self.ax_zoom.plot(x_data, y_data, label=self.df.iloc[:,col].name)

            # self.ax_zoom.plot(self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.time_col],self.df.iloc[np.arange(self.sub_min,self.sub_max,self.zoom_int),self.p_cols],label = self.df.iloc[:,self.p_cols].columns)
            self.ax_zoom.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            if(self.dive_max < self.sub_max and self.dive_max > self.sub_min):
                self.vline4 = self.ax_zoom.axvline(x = self.dive_max_ipick,color = 'blue')
            if(self.dive_min < self.sub_max and self.dive_min > self.sub_min):
                self.vline5 = self.ax_zoom.axvline(x = self.dive_min_ipick,color = 'red')
            self.figure_zoom.canvas.draw()

            self.vline1.remove()
            self.vline1 = self.ax_main.axvline(x = self.df.iloc[self.sub_min,self.time_col],color = 'red')
    ##                    figure_main.canvas.draw()
            self.vline2.remove()
            self.vline2 = self.ax_main.axvline(x = self.df.iloc[self.sub_max,self.time_col],color = 'red')
            self.figure_main.canvas.draw()

        #Redraw the audio plot
        # self.aud_min = (np.abs(self.audio.loc[:,"Timestamp"] - self.df.loc[self.sub_min,"Timestamp"])).argmin()
        # self.aud_max = (np.abs(self.audio.loc[:,"Timestamp"] - self.df.loc[self.sub_max,"Timestamp"])).argmin()
        #Update Audio plot
        if self.audio_present == True:
            self.sub_window = self.sub_max - self.sub_min
            self.aud_window = int((self.sub_window*640)/2)
            self.aud_point = int(self.frame*640)
            if self.frame > 1:
                self.aud_min = self.aud_point - self.aud_window
            else:
                self.aud_min = self.aud_point
            self.aud_max = self.aud_point + self.aud_window
            self.ax_audio.cla()
            self.ax_audio.plot(self.audio.iloc[np.arange(self.aud_min,self.aud_max,10),1],self.audio.iloc[np.arange(self.aud_min,self.aud_max,10),0])
            # self.vline_aud = self.ax_audio.axvline(x = self.dive_min_ipick,color = 'red')
            self.figure_audio.canvas.draw()
            # self.ax_audio.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))


    def audio_scroll():
        pass

    def audio_zoom():
        pass
