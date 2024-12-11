# VIP

To run scripts for given assignment

`cd ./assignment<X>`

and then run

`python main.py` + `your_file.mat` + `additional parameters`

ex. To run the script on `Beethoven.mat`, you should execute:

`python main.py Beethoven.mat`

Additional Parameters:

1. Smoothing
-s n
or
--smooth n

where n is the number of iterations of the smoothing algorithm. n is by default 100. 


2. Threshold

-t n or --threshold n

Where n is the threshold value for ransac. If the threshold value is specified, RANSAC will run. Otherwise, the default is to run Woodham approximation

