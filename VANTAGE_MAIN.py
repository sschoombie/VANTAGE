import tkinter as tk

import matplotlib
matplotlib.use('TkAgg')

import sys

import platform
if platform.system() == 'Windows':
    print("Windows")
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)

#These are the custom scripts used to drive the programme.
#They should be in the same folder as this (MAIN) script.
from Data_Frame import Data_Frame, DebugWindow

# from Functions_AVI_Test import *#Menu_functions, Menu_functions_FILE
from VANTAGE_FUNCTIONS import *#Menu_functions, Menu_functions_FILE

#Version number
vnum = "_0.01"

class App(tk.Tk):
    def __init__(self):


        ####################################################
        ## 1.  Create the main window and give it a name       ##
        ####################################################
        super().__init__()
        self.title('VANTAGE v'+vnum)
        self.state('zoomed') #This makes sure the program is loaded in full screen
        #Or you can set the desired start size
        # self.geometry("%dx%d" % (self.winfo_screenwidth() , self.winfo_screenheight()))

        #########################################
        ## 2.  Initiate the data object        ##
        #########################################
        #This creates an object with all the neccesary presets
        IMU_dat = Data_Frame()

        #########################################
        ## 3. Configure the window layout      ##
        #########################################
        #Divide the window into four blocks - with the bottom rows 1/5 the height of the top rows.
        #The bottom rows will contain information and controls while the top rows are the displays.
        #The top left block (row = 0, col = 0) is where the data is plotted
        #The top right block (row 0, col = 1) is where the video is plotted
        self.grid_columnconfigure(0,weight=1)
        self.grid_columnconfigure(1,weight=1)
        self.grid_rowconfigure(0,weight=4)
        self.grid_rowconfigure(1,weight=1)

        #Next we assign the four blocks to the data we assigned above (IMU_dat)
        #This allows us to later plot to these blocks

        #Block 1 (row 0, col 0) - Data
        IMU_dat.data_frame = tk.Frame(self,borderwidth=2, relief="solid")
        IMU_dat.data_frame.grid(column = 0,row=0,sticky = 'nesw')
        IMU_dat.data_frame.grid_propagate(False)  # Prevents resizing based on content
        IMU_dat.data_frame.grid_columnconfigure(0,weight=1)
        IMU_dat.data_frame.grid_columnconfigure(1,weight=1)
        IMU_dat.data_frame.grid_rowconfigure(0,weight=1)
        IMU_dat.data_frame.grid_rowconfigure(1,weight=1)

        #Block 2 (row 0, col 1) - Video
        IMU_dat.video_frame = tk.Frame(self,borderwidth=2, relief="solid")
        IMU_dat.video_frame.grid(column = 1,row=0,sticky = 'nesw')
        IMU_dat.video_frame.grid_propagate(False)  # Prevents resizing based on content

        #Block 3 (row 1, col 0) - Control buttons
        IMU_dat.control_frame = tk.Frame(self,borderwidth=2, relief="solid")
        IMU_dat.control_frame.grid(column = 0,row=1,sticky = 'nesw')
        IMU_dat.control_frame.grid_propagate(False)  # Prevents resizing based on content
        IMU_dat.control_frame.grid_columnconfigure(0,weight=1)
        IMU_dat.control_frame.grid_columnconfigure(1,weight=1)
        IMU_dat.control_frame.grid_columnconfigure(2,weight=1)
        IMU_dat.control_frame.grid_columnconfigure(3,weight=4)
        IMU_dat.control_frame.grid_rowconfigure(0,weight=1)
        IMU_dat.control_frame.grid_rowconfigure(1,weight=1)
        IMU_dat.control_frame.grid_rowconfigure(2,weight=1)
        IMU_dat.control_frame.grid_rowconfigure(3,weight=1)

        #Block 4 (row 1, col 1) - Debugging output
        IMU_dat.debug_frame = tk.Frame(self,borderwidth=2, relief="solid")
        IMU_dat.debug_frame.grid(column = 1,row=1,sticky = 'nesw')
        IMU_dat.debug_frame.grid_propagate(False)  # Prevents resizing based on content

        ##########################
        ## 4.  Setup Debug window #
        ##########################
        #We relay everyting that is printed to the debug window in Block 4
        t = tk.Text(IMU_dat.debug_frame)
        # t.pack()
        t.place(relx = 0,rely = 0,relwidth = 1, relheight =1)
        debug_w =  DebugWindow(t)
        sys.stdout = debug_w
        print("Open file to start")

        ###########################################################
        ## 5.  Create the menu items and assing functions to them  #
        ###########################################################
        #Add a function to the closure of the window
        self.protocol('WM_DELETE_WINDOW', lambda:Menu_functions_FILE.f_quit(self,IMU_dat))

        #Add a function to monitor key presses
        self.bind("<KeyPress>",lambda event:Keyboard_functions.key_press(event,self,IMU_dat))

        #Initiate the menu
        menu = tk.Menu(self)

        #############
        # File menu #
        #############
        IMU_dat.filemenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=IMU_dat.filemenu)

        IMU_dat.filemenu.add_command(label="Load csv...", command=lambda:Menu_functions_FILE.load_csv(self,IMU_dat))
        IMU_dat.filemenu.add_command(label="View data", command=lambda:Menu_functions_FILE.view_data(self,IMU_dat))
        # IMU_dat.filemenu.add_command(label="Load gps...", command=lambda:Menu_functions_FILE.load_gps(IMU_dat))
        # IMU_dat.filemenu.add_command(label="Import config", command=lambda:Menu_functions_FILE.import_config(IMU_dat))

        IMU_dat.filemenu.add_separator()
        IMU_dat.filemenu.add_command(label="Exit", command=lambda:Menu_functions_FILE.f_quit(self,IMU_dat))


        #############
        # Video menu #
        #############
        videomenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Video", menu=videomenu)
        videomenu.add_command(label="Load video...", command=lambda:Menu_functions_VIDEO.load_avi(self,IMU_dat))
        videomenu.add_command(label="Set video directory...", command=lambda:Menu_functions_VIDEO.save_wd(IMU_dat))
        videomenu.add_command(label = "Convert video to mp4", command = lambda: Menu_functions_VIDEO.convert_video(self,IMU_dat))
        videomenu.add_command(label="Load wav...", command=lambda:Menu_functions_VIDEO.load_audio(self,IMU_dat))
        ###############
        # Filter menu #
        ###############
        # filtermenu = tk.Menu(menu,tearoff = 0)
        # menu.add_cascade(label = "Filters", menu = filtermenu)
        #
        # filtermenu.add_command(label = "LULU Up", command = lambda:Menu_functions_FILTER.apply_filter(self,IMU_dat,0)) #LULU up
        # filtermenu.add_command(label = "LULU Down", command = lambda:Menu_functions_FILTER.apply_filter(self,IMU_dat,1)) #LULU down
        # filtermenu.add_command(label = "Gradient", command = lambda:Menu_functions_FILTER.apply_filter(self,IMU_dat,2)) #Gradient
        # filtermenu.add_command(label = "Roll mean", command = lambda:Menu_functions_FILTER.apply_filter(self,IMU_dat,3)) #Rolling mean

        #############
        # Plot menu #
        #############
        plotmenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Plot", menu = plotmenu)

        plotmenu.add_command(label="Choose axes", command=lambda:Menu_functions_PLOT.choose_axes(self,IMU_dat))
        plotmenu.add_command(label="Plot data", command=lambda:Menu_functions_PLOT.plot_dat(self,IMU_dat))

        # plotmenu.add_command(label = "Spectrogram", command = lambda:Menu_functions_PLOT.plot_spec(self,IMU_dat)) #Spectrogram
        # plotmenu.add_command(label = "Notch filter", command = lambda:Menu_functions_PLOT.notch_filter(IMU_dat)) #Notch filter
        # plotmenu.add_command(label = "Audio", command = lambda:Menu_functions_PLOT.plot_audio(self,IMU_dat)) #Notch filter

        #################
        # FUnctions menu #
        #################

        analysismenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Functions", menu = analysismenu)
        analysismenu.add_command(label = "Calculate ACC metrics", command = lambda:Menu_functions_ANALYSIS.acc_metrics(self,IMU_dat)) #Calculate accelerometer metrics (VeDBA etc.)
        analysismenu.add_command(label = "Find dives", command = lambda:Menu_functions_ANALYSIS.find_dives(self,IMU_dat)) #Find dives automatic method
        analysismenu.add_command(label = "Video squash", command = lambda:Menu_functions_ANALYSIS.squash_vid(self,IMU_dat)) #Video Squash
        analysismenu.add_command(label = "Sync dives (Manual)", command = lambda:Menu_functions_ANALYSIS.sync_dives_manual(self,IMU_dat)) #Find dives manual method
        analysismenu.add_command(label = "Sync dives (AUTO)", command = lambda:Menu_functions_ANALYSIS.synch_dives_auto(self,IMU_dat)) #Find dives manual method
        analysismenu.add_command(label = "Navigate events", command = lambda:Menu_functions_ANALYSIS.pce_navigate(self,IMU_dat)) #Export ALL PCE images

        #add a submenu under analysis menu
        analysis_submenu = tk.Menu(analysismenu,tearoff = 0)
        analysismenu.add_cascade(label = "CV filters",menu = analysis_submenu)
        analysis_submenu.add_checkbutton(label = "Horizon detect",variable = IMU_dat.horison_detect, command = lambda:Menu_functions_ANALYSIS.check_horison_column(self,IMU_dat))

        ###################
        # Annotate menu #
        ###################
        annotatemenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Annotate", menu = annotatemenu)
        #Add a checkbotton to see if
        annotatemenu.add_command(label = "Annotation column", command = lambda:Menu_functions_ANNOTATE.choose_annotation_col(self,IMU_dat)) #Video Squash
        annotatemenu.add_checkbutton(label="Annotate selection", variable=IMU_dat.annotate_selection, command = lambda:Menu_functions_ANNOTATE.selection_warning(self,IMU_dat))

        # annotatemenu.add_command(label="Export events", command=lambda:Menu_functions_EXPORT.export_events(self,IMU_dat))



        ###################
        # Export menu #
        ###################
        pcemenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Export", menu = pcemenu)
        pcemenu.add_command(label="Export events", command=lambda:Menu_functions_EXPORT.export_events(self,IMU_dat))
        pcemenu.add_command(label="Export config", command=lambda:Menu_functions_EXPORT.export_config(IMU_dat))
        pcemenu.add_command(label = "Export annotated images", command = lambda:Menu_functions_EXPORT.export_annotated(self,IMU_dat)) #Export ALL PCE images
        pcemenu.add_command(label = "Export YOLO images", command = lambda:Menu_functions_EXPORT.export_yolo(self,IMU_dat)) #Export ALL PCE images
        # pcemenu.add_command(label = "Export dive images", command = lambda:Menu_functions_EXPORT.export_dive_images(self,IMU_dat)) #Export ALL PCE images

        # #########
        # # Model #
        # #########
        modelmenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Model", menu = modelmenu)
        modelmenu.add_command(label = "Load YOLO", command = lambda:Menu_functions_MODEL.model_YOLO(self,IMU_dat)) #Load a pre-trained YOLO model and its weights
        modelmenu.add_command(label = "Predict with YOLO", command = lambda:Menu_functions_MODEL.run_YOLO(self,IMU_dat)) #Load a pre-trained YOLO model and its weights
        #
        # modelmenu.add_command(label = "Load model", command = lambda:Menu_functions_MODEL.model_load(self,IMU_dat)) #Load a pre-trained model and its weights
        # modelmenu.add_command(label = "Preprocess data", command = lambda:Menu_functions_MODEL.model_pre_process(self,IMU_dat)) #Pre-process the data to conform to the model parameters
        # modelmenu.add_command(label = "Predict PCE", command = lambda:Menu_functions_MODEL.model_predict(self,IMU_dat)) #Predict PCE from model
        # modelmenu.add_command(label = "Plot results", command = lambda:Menu_functions_MODEL.model_results(self,IMU_dat)) #Plot the output (if annotated)

        ###################
        # Cheatsheat menu #
        ###################
        plotmenu = tk.Menu(menu,tearoff = 0)
        menu.add_cascade(label = "Help", menu = plotmenu)

        plotmenu.add_command(label = "Load data", command = lambda:Menu_functions_CHEATSHEETS.analysis_seq(self,IMU_dat)) #Find dives automatic method
        plotmenu.add_command(label = "Video controls", command = lambda:Menu_functions_CHEATSHEETS.vid_control(self,IMU_dat)) #Video Squash
        plotmenu.add_command(label = "View data", command = lambda:Menu_functions_CHEATSHEETS.view_data(self,IMU_dat)) #Video Squash


        #Assign the menus to the main window
        self.config(menu=menu)

        ###########################################################
        ## 6.  Create the control panel with buttons and sliders   #
        ###########################################################



        #Buttons
        btn_prev = tk.Button(IMU_dat.control_frame,text = "Previous video",command=lambda:Button_functions.prev_vid(self,IMU_dat))
        btn_prev.grid(column = 0,row=0,padx = 5, pady = 5,sticky = "nesw")

        btn_next = tk.Button(IMU_dat.control_frame,text = "Next video",command=lambda:Button_functions.next_vid(self,IMU_dat))
        # btn_next.place(relx = 0.9,rely = 0)
        btn_next.grid(column = 1,row=0,padx = 5, pady = 5,sticky = "nesw")

        btn_rst = tk.Button(IMU_dat.control_frame,text = "Reset video",command=lambda:Button_functions.rst_video(IMU_dat))
        btn_rst.grid(column = 2,row=0,padx = 50, pady = 5,sticky = "nesw")

        btn_photo = tk.Button(IMU_dat.control_frame,text = "Save image",command=lambda:Button_functions.save_image(IMU_dat))
        btn_photo.grid(column = 0,row=1,padx = 5, pady = 5,sticky = "nesw")

        btn_vid = tk.Button(IMU_dat.control_frame,text = "Save video clip",command=lambda:Button_functions.save_vid_clip(IMU_dat))
        btn_vid.grid(column = 1,row=1,padx = 5, pady = 5,sticky = "nesw")

        # btn_res = tk.Button(IMU_dat.control_frame,text = "Set Resolution",command=lambda:Button_functions.set_res(IMU_dat))
        # btn_res.grid(column = 0,row=2,padx = 5, pady = 20,sticky = "nesw")

        # btn_update = tk.Button(IMU_dat.control_frame,text = "Update Frame",command=lambda:Button_functions.update_frame(IMU_dat))
        # btn_update.grid(column = 1,row=2,padx = 5, pady = 20,sticky = "nesw")

        # btn_smooth = tk.Button(IMU_dat.control_frame,text = "Apply smooth",command=lambda:Button_functions.apply_smooth(IMU_dat))
        # btn_smooth.grid(column = 0,row=3,padx = 5, pady = 5,sticky = "nesw")

        #Sliders
        IMU_dat.speed_slider = tk.Scale(IMU_dat.control_frame, from_=1, to=100, orient="horizontal",label = "Speed (fps)")
        IMU_dat.speed_slider.set(24)
        IMU_dat.speed_slider.grid(column = 3,row=1,padx = 5, pady = 5,sticky = "nesw")

        # IMU_dat.res_slider = tk.Scale(IMU_dat.control_frame, from_=480, to=1440, orient="horizontal",label = "Resolution")
        # IMU_dat.res_slider.grid(column = 3,row=2,padx = 5, pady = 5,sticky = "nesw")

        # IMU_dat.slider2 = tk.Scale(IMU_dat.control_frame, from_=1, to=10, orient="horizontal",label = "smooth")
        # IMU_dat.slider2.grid(column = 3,row=3,padx = 5, pady = 5,sticky = "nesw")

        #Check boxes
#         c1 = tk.Checkbutton(self, text='Img blur',variable=img_blur, onvalue=1, offvalue=0 )
# ##        c1.place(x = 500,y = 3)
#         c1.place(relx = 0.75,rely = 0)
#         c2 = tk.Checkbutton(self, text='Img thresh',variable=img_th, onvalue=1, offvalue=0)
# ##        c2.place(x = 500,y = 23)
#         c2.place(relx = 0.75,rely =0.04)
#         c3 = tk.Checkbutton(self, text='Img edge',variable=img_edge, onvalue=1, offvalue=0)
# ##        c3.place(x = 500,y = 43)
#         c3.place(relx = 0.75,rely = 0.08)

        c3 = tk.Checkbutton(IMU_dat.control_frame, text='Brighten',variable=IMU_dat.img_bright, onvalue=1, offvalue=0)
##        c3.place(x = 500,y = 43)
        # c3.place(relx = 0.65,rely = 0.0)
        c3.grid(column = 3,row=2,padx = 5, pady = 5,sticky = "nesw")
        # tk.Checkbutton(self, text="Normalize?",variable=IMU_dat.norm, onvalue=1, offvalue=0,command =lambda:Button_functions.normalize(IMU_dat) ).place(relx = 0.75,rely = 0.16)



#Run the app
if __name__ == '__main__':
    app = App()
    app.mainloop()
