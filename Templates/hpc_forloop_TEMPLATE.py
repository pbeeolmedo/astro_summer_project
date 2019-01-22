#---- Imports
import multiprocessing
import joblib

#---- A few things which need to be specified + add folder paths etc
forloop_list = [] #define the list for iterations in for-loop
numChuncks = 1 # split list into x chuncks
numOfCoresToUtilise = 1 # define how many cores you want to use

# ---- Some useful info to be printed
numIters = len(forloop_list)
print(f"Number of iterations in loop is : {numIters}")
num_cores = multiprocessing.cpu_count()
print(f"Number of cores on computer: {num_cores}")

# ---- THE MAIN PART: FOR-LOOP (to be placed inside for-loop in function)
def forloopFunc(inputs=[start,end])
    for i in forloop_list[start,end]:
        #do some things
    return #something

# ---- Just a simple function to help w splitting
def list_index_splitter(length_list,chunks=1):
    inputs = []
    print('hello')
    len_chunks = int(length_list/chunks)
    for i in range(chunks):
        start = i*(len_chunks)
        if i == chunks-1:
            end = -1
        else:
            end = (i+1)*(len_chunks)-1
        inputs.append([start,end])
    return inputs
inputs = list_index_splitter(numFiles,numChuncks)
print(f"Inputs are : {inputs}")

# ---- Executing the for loop over the specified cores given the list chuncks to go through
Parallel(n_jobs=numOfCoresToUtilise)(delayed(forloopFunc)(i) for i in inputs)
