# FUNCTIONS found in the <i>Functions</i> menu

## Calculate ACC metrics
This will calculate commonly used accelerometer metrics from raw tri-axial accelerometer data, where the user will be prompted to choose the correct columns.
Once the process is completed, the computed metrics (e.g. roll, pitch, vedba, vesba, odba) will be available from the <i> Plot - Choose axes </i> menu.

## Find dives
Where depth data is available this function will calculate dive metrics for individual dives, the user will be prompted to choose the correct column to use.
This function may take a long time to run, depending on the size of the data set and the processing power of the user computer.
The dive data is saved as a CSV file with the same name as the loaded file but wit a suffix "_DIVES". If the _DIVES file is present it will be loaded instead of rerunning the analyses.

## Video squash
This will produce a squashed image (or barcode image) from each video clip in the chosen video directory.
These images are created by extracting the average pixel colour value from each frame in the video clip and stitching it as a single image. This creates a single image, 
where each width pixel (horizontal) represents a frame from the video.
These images are saved within the video directory and subsequently loaded with the video clips, shown horizontally under the video frame.

## SYNC dives
<b> NB! Prerequisites 
- <i> Functions - Find dives </i> should be run
- Sqaushed image (barcode image) should be loaded
</b>

Dives times can be used to synchronize time between video and time-series data in a semi-autonomous, or autonomous way.
We assume that three successive deep dives will not have the same duration, and/or the period between the start times of the dives won't be the same.

<b> If successful this will synchronize the time frames to the resolution of the depth data (often 1 Hz) - the synchronization can then be manually adjusted to attain 
synchronization at higher frequencies, i.e. 25 Hz to match with accelerometer data.</b>

<b> IMPORTANT! The user should always double-check if the synchronization makes sense before continuing with futher analyses. </b>

### Sync dives (MANUAL)
These steps are done by finding dives from the VIDEO DATA ONLY. The user finds dives in the video and VANTAGE matches the selected dives to the time-series data.

A control panel will appear allowing the user to select the start and end frames (from the video data) for three successive dives. This will work best if three successive DEEP dives (> 20 m)
lasting at least 30 s is used. This will be easy to see from the squashed image (see below figure). 

![VANTAGE_SYNC_MANUAL_GitHub](https://github.com/sschoombie/VANTAGE/assets/49139080/22146231-774b-4544-ac83-2804765d1634)

To do this, navigate through the video using the keyboard controls to play the video, or using the scrollbar.
Navigate to the frame where the animal enters the water for the first dive and press the <i> Dive 1 START </i> button, then navigate to the frame where the animal exits the water after the dive
and press the <i> Dive 1 END </i> button. Do the same for the following two dives.
Once the start/end frames for three dives have been identified, press the <i> Video SYNC </i> button. 
A prompt will let you know if the synchronization was successful.

### Sync dives (AUTOMATIC)
The user chooses a video clip where at least three successive deep (> ~20 m) long (> ~30 s) dives are present.
Then <i> Functions - Sync dives (AUTO) </i> is used. This will automatically detect the start times of dives from the sqaushed image (barcode image) of the video,
and find the matching pattern in the dive data, allowing the calculation of the temporal offset between the video and the time-series data.

## Navigate events

Once annotations have been made, the user can navigate to the previous or next available annotation. This enables quick navigation between annotated points.
If the next annotation is in a new video clip, the relevant clip will be loaded automatically.

## CV filters
Computer vision filters

### Horizon detect
This function is applicable to flying seabirds, where an unobstructed horizon is visible.
This will automatically detect the horizon and display the estimated angle in the video.
