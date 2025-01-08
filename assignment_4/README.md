# Image Retrieval Experiment

To run scripts for the image retrieval experiments:

`cd ./assignment_4

and then run:

`python main.py` + parameters

## Example:
To run experiment 1 with 20 categories and load existing data, execute:
```bash
python main.py -e 1 -c 20 -l
```

## Parameters:

1. **Experiment Number**  
   `-e n` or `--experiment n`  
   Where `n` is the experiment number. Options:
   - `1`: Runs the retrieval experiment on the training data.
   - `2`: Runs the retrieval experiment on the testing data.

2. **Number of Categories**  
   `-c n` or `--categories n`  
   Where `n` is the number of categories to use. Options:
   - `5`: Use 5 selected categories.
   - `20`: Use 20 selected categories.

3. **Load Existing Data Files (optional)**  
   `-l` or `--load`  
   If specified, the script will use existing data files instead of generating new ones. This is optional; without it, the script will generate new data.

## Output:
The script will:
1. Save evaluation metrics (Mean Reciprocal Rank, Top-3 Accuracy, Classification Accuracy) as bar plots in the `results/` directory.
2. Print detailed metrics per category and overall to the terminal.

The Mean Reciprocal Rank and Top-3 Accuracy are calculated as described in the assignment pfd. The Classification is calculated as the most common category out of the top 5 search results.

## Additional Notes:
To ensure proper setup, verify the required dependencies (`OpenCV`, `NumPy`, etc.) are installed.