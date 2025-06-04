Project Authors: Petar Miljkovic, Philippe Justin Clark, Alexander Blom Tindlund, Finn St¨ablein

We have included in our repository multiple scripts for our project, including:
1. *baseline.py*: The script containing our baseline implementation.
2. *original.py*: The script that contains our original model, where, rather than using LSTM directly, we encoded the timeseries data and ran it through a feed-forward neural network.
3. *intermediate_LSTM.py*: The script with our original LSTM model implementation.
4. *PCA_LSTM.py*: The script for our final project implementation using PCA.

All of these scripts, except *original.py*, can be run from the command line or via PyCharm, the IDE used in development. Before running any individual script, please note that it must be created and executed within the *cs221_proj* environment specified in the *environment.yml* file. 

Note: Our original FFN model was set up using a different data pipeline that was cumbersome to update and maintain. Therefore, this script currently will not run as is. This model is included solely for our own reference, in case we wish to revisit older methods in the future.

Also included are plots from our final runs using the PCA variation of the script, with hyperparameters set to the same settings.

Training was done using an NVIDIA GeForce RTX 4060 Ti card. 

