# VANTAGE (Visualization and analysis of video and time-series data from animal-borne loggers)

VANTAGE is a graphical user interface written in Python that enables users to simultaneosly view VIDEO and TIME-SERIES data. 
For more information please see the ![journal article](https://doi.org/10.1111/2041-210X.70026) published in Methods in Ecology and Evolution.

The main function of VANTAGE is for temporal synchronization of time-series and video data with concurrent visualization and navigation, which allows effecient labelling of the data.

The software was developed for use with data collected from krill-feeding penguins, and provide additional functions for such data. However, the modular design of the software allows for additional functions to be added (or modification of existing functions) applicable to specific species.

![VANTAGE_example_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/e9be375d-1d1c-4ee8-b3dc-e74a0e1a0d43)


## 1. Installation 
A couple of python libraries have to be installed for VANTAGE to run.
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following:

### OpenCV for video handling
The latest version of OpenCV has bug when reading the frame rate - so we load a previous stable version
```bash
pip install opencv-contrib-python==4.7.0.72
```
### Pandas for data frames
```bash
pip install pandas
```
### Matplotlib and pillow for plotting
```bash
pip install matplotlib
```
```bash
pip install pillow
```
### Optional - SciPy for machine learning applications
Installation of scipy and ultralytics is only neccesary if machine learning applications are used
```bash
pip install scipy
```
```bash
pip install ultralytics
```
### Optional - FFmpeg for video conversion
Installation of FFmpeg is only neccesary if you need to convert videos to mp4
1. Download the latest version of [FFmpeg](https://ffmpeg.org/download.html)
2. See these guides for installation on [Windows](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/), [Linux](https://www.redswitches.com/blog/install-ffmpeg-on-ubuntu-and-linux/), or [Mac](https://phoenixnap.com/kb/ffmpeg-mac)

## 2. Running VANTAGE
Make sure the following files are downloaded in the same directory:
- Data_Frame.py
- VANTAGE_FUNCTIONS.py
- VANTAGE_MAIN.py

open command prompt and type:
```bash
python VANTAGE_MAIN.py
```
OR (for newer versions of Python)
```bash
py VANTAGE_MAIN.py
```

This will open the following window:
![VANTAGE_start_screen_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/a315a6cd-8068-4fcd-8a1f-4f6792864c68)


### 2.1 Loading TIME-SERIES data

<i>File - Load csv...</i>

![VANTAGE_load_csv_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/7acb7230-1783-47c3-8cdd-a4f642767952)

If the selected .csv file has a column named "Timestamp" the data will load automatically - otherwise a prompt will appear to select the column where the date and time is located.
When successfully loaded a plot will appear showing the data.


### 2.2 Loading VIDEO data

#### 2.2.1 First select the folder where the video files are located


<i>Video - Set video directory...</i>


![VANTAGE_set_video_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/a9ea123d-0cc7-4a41-977c-34a6d35c0869)


#### 2.2.2 VERY IMPORTANT! If the video files are not .mp4 it should be converted


<i>Video - Convert video to mp4</i>


![VANTAGE_Convert_video_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/63c02fe3-4f76-4625-9eb2-1edab149117d)

This will convert the selected files to .mp4 at 25 frames per second while preserving the original timestamp.


#### 2.2.3 Next select the video file to load


<i>Video - Load video...</i>


![VANTAGE_load_video_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/476a05c5-6e9b-4528-b246-5f19b0b5692a)


### If everything loaded correctly you will see the following screen:


![VANTAGE_working_screen_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/84842d34-4903-4ee1-b242-17943c80b119)


### Please see the files in the "Tutorials" folder for specific uses, which include (but not limited to):
- Time synchronization
- Squashed (or barcode) images from video
- YOLO model predictions

 ![Figure 5 - YOLO](https://github.com/sschoombie/VANTAGE/assets/49139080/d3ec5742-db52-4c7b-9e85-f961da9d50d6)

 - Horizon detection for flying seabirds

![VANTAGE_horizon_GitHub](https://github.com/user-attachments/assets/d16415ee-1b6f-403c-86b9-98a99c4d38d0)




