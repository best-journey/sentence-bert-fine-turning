# how to use this fine-turing pre-trained model  
## step one
Due to the limitation of one file size which must less than 25 MB in github repository. we choose to cut the big model file into many small files for avoiding this problem.
So when you git clone the repository from remote end into the local end, the first thing you need to do is to merge the small files into one big file by executing Python script named initialize.py. 
Note that you should promise that the all files in local reposity are not opened by file explorer, or the process of executing Python script named initialize.py will generate a error deleting small files with a failure.
## step two 
To execute the Python scripts which are load_model_after_fine_turning.py and load_model_before_fine_turning.py is the usage of this pre-trained model.
