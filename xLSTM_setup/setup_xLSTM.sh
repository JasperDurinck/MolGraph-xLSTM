#!/bin/bash
pip3 install ninja

#the sLSTM compiling will have issue with  >= 9 =< 13, but note that even though is says gcc in the error,
#after updating, you also need the compatible gxx to stop the same error

##Note that for python 3.10 there is no gcc or g++ version dependency issue (unlike for 3.12)
#conda install -c conda-forge gcc=11.4.0 #(note this can take a while) #for xLSTM we need >= 9 =< 13, tested 11.4 locally and that should work 
#conda install conda-forge::gxx=11.4.0

#fatal error: cuda_runtime_api.h: No such file or directory # this maybe due to not having this file system side or issue with xLSTM command prompt generation for env based cuda toolkits 
conda install -c nvidia cuda-toolkit=12.1
