Clean WildChat: 
The code loads the WildChat dataset, filters it to only English conversations, deeply cleans and normalizes every message, and saves the processed results for later use. 

O*Net preprocessing:
This script reads a list of tasks from O*Net task statement Excel, cleans and standardizes the task text for NLP, and saves the results for further analysis.

Data preprocessing:
The code splits conversation into prompt and response column, convert them to data frame. It converts all text to lowercase, removes stop word, punctuation, special characters, preforms lemmatization, and turn TF-IDF vectorized matrix for further clustering.

Filter work samples:
The function of this code is to filter work-related and non work-related samples from the whole English dataset. Install required packages and load the dataset to run the code. 

Remove line terminators:
Remove line terminators for prompt and response pairs. 

WildChat conversation reconstruction:
Reconstruct prompts and responses into a piece of conversation for all samples. 