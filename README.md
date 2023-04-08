# Thresholded Linear Bandits

Code is used with python 3.10.6. Libraries used are: math, numpy, matplotlib.pyplot, multiprocessing, and timeit. These libraries are imported at the start of both codes.

The code is split into two files, main.py and Graphing.ipynb. main.py needs to be run first, then Graphing.ipynb can be run once the results of main.py are generated.

main.py is the code for the one-dimensional algorithms from the paper and runs the four experiments: varying Delta with a static tau, and varying tau with a static Delta, varying T in the Delta> tau case, and varying T in the tau > Delta case. This outputs 12 .csv files with the experimental results. These .csv files will be used in Graphs.py. To run:

python main.py

Graphing.ipynb must be located in the same file as the saved outputs from main.py. This code will generate the plots used in the paper. If you changed the variables x or reps from the experimental setup in main.py, then you must also change them in Graphing.ipynb to match before running. Run with Jupyter notebook.
