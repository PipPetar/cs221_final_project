We have included in our repository multiple scripts for out project, including:
1. *baseline.py* : The script with our baseline implementation.
2. *original.py* : The script that contains our original model, where rather than using LSTM directly we encoded timeseries data and rad it through a feed-forward neural network.
3. *intermediate_LSTM.py* : The script with our original LSTM model implementation.
4. *PCA_LSTM.py* : The script with our final project implementation using PCA.

All of these scripts, except *original.py*, can be run from the command line or via PyCharm, the IDE used in development. Prior to running any individual script, note that one must create and run via the *cs221_proj* environment specified in the *environment.yml* file.

Also included are plots from our final runs using the PCA variation of the script, with hyperparameters set to the same settings.

Training was done using an NVIDIA GeForce RTX 4060 Ti card. 
