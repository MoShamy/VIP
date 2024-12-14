#!/bin/bash

# Run all test cases
python main.py Beethoven -i

python main.py mat_vase -i

# Default parameters
python main.py shiny_vase -i

# Using RANSAC
python main.py shiny_vase -t 2.0 -i

# Then smoothing as well
python main.py shiny_vase -t 2.0 -s 100 -i

# shiny_vase2
python main.py shiny_vase2 -i

python main.py shiny_vase2 -t 2.0 -i

python main.py shiny_vase2 -t 2.0 -s 100 -i

# Buddha
python main.py Buddha -i

python main.py Buddha -t 25.0 -i

python main.py Buddha -t 25.0 -s 100 -i

# face
python main.py face -t 10.0 -i

python main.py face -t 10.0 -s 100 -i
