# Thresholded Linear Bandits

Code is used with python 3.10.6. Libraries used are: math, numpy, matplotlib.pyplot, multiprocessing, and timeit. These libraries are imported at the start of both codes.

The code is split into three files, main.py , main_graphing.ipynb, and graphing.ipynb. main.py needs to be run first, then main_graphing.ipynb and graphing.ipynb can be run once the results of main.py are generated.

main.py is the code for the one-dimensional algorithms from the paper and runs the four experiments: varying Delta with a static tau, and varying tau with a static Delta, varying T in the Delta> tau case, and varying T in the tau > Delta case. This outputs 12 .csv files with the experimental results. To run:

python main.py

main_graphing.ipynb graphing.ipynb must be located in the same file as the saved outputs from main.py. The data files have been compress so please ensure the paths are correct before running the notebooks. main_graphing generate the plots used in the main paper and graphing.ipynb will generate the plots used in the appendix. If you changed the variables x or reps from the experimental setup in main.py, then you must also change them in notebooks to match before running. Run with Jupyter notebook.
