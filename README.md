# Automated-nucleus-morphological-change-detection_

## Table of Contents

- [Copy the repository](#copy-the-repository)
- [Installation](#installation)
- [Using the code](#using-the-code)


## Copy the repository
1) Click on the green button <img width="947" height="222" alt="image" src="https://github.com/user-attachments/assets/670bc17e-1c0e-402a-8f85-c51a227e6fb8" />
2) Copy the HTTPS link <img width="530" height="334" alt="image" src="https://github.com/user-attachments/assets/0f8d92ea-e38b-4eb0-a6bc-951ef7779062" />
3) Create a folder in your files (it is where the code will be located) and go in this folder
4) Go to the terminal (right click then "open in terminal") <img width="743" height="477" alt="image" src="https://github.com/user-attachments/assets/482dbf62-50d8-4da5-b2ff-2e9129bd266a" />
5) Write in the terminal "git clone + the link" and press enter <img width="1093" height="68" alt="image" src="https://github.com/user-attachments/assets/a675447a-9b2b-4398-a3fc-2fd7fd981fe1" />
6) All the code is now in this folder
   

## Installation

Create and activate a conda environment with the required dependencies:

1) Go to the app "Anaconda Prompt"
2) Write in the terminal: ``` conda create -n features-detection --file conda-requirements.txt -c conda-forge -c pytorch ```
3) When the environment is created: ``` conda activate features-detection ```

## Using the code 
1) Open VScode
2) Open the folder containing the code. The files opened should look like this <img width="299" height="125" alt="image" src="https://github.com/user-attachments/assets/0a966ce6-b9c0-496b-ab86-8fcb8f89be1e" />
# Segmentation
1) To use the segmentation, go to /segmenter/main.py
2) Click on the arrow on the top right of the page <img width="181" height="80" alt="image" src="https://github.com/user-attachments/assets/35316015-e074-4700-a517-c978d03cfc42" />
3) Select the folder that contains folders of your images
4) Select the folder where you want the results to be stored (the names of the results will be automatically determined from the input names)
5) Let the code run (it can take a bit of time if you have a lot of images)
# Comparaison

1) To compare 2 dataframes, go to comparaison/main_2groups.py
2) Run the script
3) Select the data frames in this order exactly: morphological features from the 1st population, DNA features from the 1st population, morphological features from the 2nd population, DNA features from 2nd population.
   









```bash
conda create -n features-detection --file conda-requirements.txt -c conda-forge -c pytorch
conda activate bio-compare
conda config --set channel_priority strict
```
