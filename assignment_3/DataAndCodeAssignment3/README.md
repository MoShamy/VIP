# VIP

To run scripts for given assignment

`cd ./assignment<X>`

and then run

`python main.py` + `your_file.mat`

ex. To run the script on `Beethoven.met`, you should execute:

python main.py Beethoven.met

Additional Parameters:

1. Smoothing
-s n
or
--smooth n

where n is the number of iterations of the smoothing algorithm. n is by default 100. 


2. Threshold

-t n or --threshold n

Where n is the threshold value for ransac. If the threshold value is specified, RANSAC will run. Otherwise, the default is to run Woodham approximation

3. Image mode
-i or --image

This will run the script in non-interactive mode and only output the matplotlib figures and .png files in the 'figures' directory.


To generate all images used in assignment you can also run `./run\_all.sh`
