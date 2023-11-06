# Installation for Ubuntu VM
## Step-by-Step Guide

If you are reading this, we suppose you have ready to run Ubuntu on VM, in your system. if not, then go to some recent youtube videos and make sure you have Ubuntu successfully running on VM.

1. Open Your Virtual Machine
2. Open Terminal (ctrl + T)
3. Clone this repo https://github.com/google/CFU-Playground by using ```git clone https://github.com/google/CFU-Playground```
4. Go to this directory "CFU-Playground/third_party/python/vizier/" using ```cd CFU-Playground/third_party/python/vizier/```
5. Now you will see that there is a CFU-Playground folder. Now go to "CFU-Playground/third_party/python/vizier/" and see if vizier folder is empty or not! ![Alt text](./docs/installation_images/file_preview.png?raw=true "Title")
10. Go to python folder using terminal (location: CFU-Playground/third_party/python/) using command like ```cd CFU-Playground/third_party/python/```
11. run ```rm -rf vizier```
13. Clone vizier repo, run  ```git clone https://github.com/ShvetankPrakash/vizier```
15. ```cd CFU-Playground/``` -> ```cd scripts/``` ->  Run setup_vizier.sh file using command ```./setup_vizier.sh```
19. Might give some errors, therefore activate conda environment/ or create one, If you have existing environments, you can find the list using this command ```conda env list``` if you don't find one, create using following commands: Open anaconda terminal and run ```conda create -n myenv```. Replace myenv with the environment name. now activate that enviroment using ```conda activate myenv```
      1. Install anaconda
      2. 
22. Now run ```./setup_vizier.sh```again
23. Some Errors might occur due to version of python. 
24. run ```sudo apt install build-essential```
25. run ```pip install cvxopt```
26. run ```export CVXOPT_BUILD_FFTW=1```
27. ```pip install cvxopt --no-binary cvxopt```
28. ```conda install -c conda-forge cvxopt```
29. ```./setup_vizier.sh```
30. Might give errors related to ale-py
31. Go to file requirements-benchmarks.txt:
32. third party -> python -> vizier -> vizier -> requirements-benchmarks.txt
33. Comment out all the lines
34. Now running setup_vizier.sh file would not give ale-py error
35. Now, try some other example - go to CFU-Playground -> proj -> dse_template -> vizier_dse.py
36. Go to line 40, comment it out and add this line
37. cycles, cells = 1, 1
38. Run this file. It should run successfully without any errors.

