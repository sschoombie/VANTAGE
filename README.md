# VANTAGE (Visualization and analysis of video and time-series data from animal-borne loggers)

VANTAGE is a graphical user interface written in Python that enables users to simultaneosly view VIDEO and TIME-SERIES data. 

## Installation 
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
```bash
pip install scipy
```


