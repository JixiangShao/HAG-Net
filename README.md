# HAG-Net
This is the implementation of HAG-Net: A Vessel Segmentation Model for Retinal Medical Images Integrating Global Perception and Graph Structure Reasoning

ENVIRONMENT

window10(Ubuntu is OK)+pycharm+python3.9+pytorch1.3.1

DATA

The DRIVE dataset is from https://drive.grand-challenge.org/

The Stare dataset is from https://cecas.clemson.edu/~ahoover/stare/

The CHASE_DB1 dataset is from https://blogs.kingston.ac.uk/retinal/chasedb1/


HOW TO RUN


First of all, you need the DRIVE database. We are not allowed to provide the data here, but you can download the DRIVE database at the official website. Extract the images to a folder, and call it "DRIVE", for example. This folder should have the following tree:
  
  DRIVE
  
  │

  └───test

  |    ├───1st_manual
  
  |    └───2nd_manual

  |    └───images

  |    └───mask
  
  │
  
  └───training

      ├───1st_manual

      
      └───images

      
      └───mask



The same procedure applies to other datasets as well. The only thing you should do is enter the dataset.py and correct the path of the datasets. 

RUSLUTS

after train and test,"results" folder will be created.

save_weights_DRIVE

After train,the saved model is in this folder.

results_DRIVE folder

in this folder,there are the ouput predict of the saved model,such as:
<img width="565" height="584" alt="image" src="https://github.com/user-attachments/assets/27cde6a3-de4c-4801-a540-cbf9febf4662" />


Detailed information will be made public after the paper is accepted.
