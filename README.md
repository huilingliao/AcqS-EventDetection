# AcqS-EventDetection
Acquisition-guided sampling approach for rare event detection.
The code is written in python and implemented through botorch. 

## Description
We propose a sampling-based strategy for rare event detection with sampling distribution determined by pre-specified acquisition function. Instead of optimizing a possibly multimodal complicated acquisition function, sampling is utilized and HMC is adopted for generating the pool of specification candidates for evaluation. Customized acquisition function and potential add-on modules including pre-screening for incorporating various criteria and reweighting or subset selection for surrogate model training can be used. 

## Installation 
To install the package in R, run the following commands: 
```{r}
install.packages("devtools")
library(devtools)
install_github("huilingliao/AcqS-EventDetection")
```

% ## Example
% ```{r}
% ```

## Author
This package is written by Huiling Liao (hliao13@iit.edu). 

## Reference 
