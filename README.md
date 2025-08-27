<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/tobibui1904/Fannie-Mae-Credit-Risk-Modelling">
  </a>

  <h3 align="center">Fannie Mae Credit Risk Modelling</h3>

<!-- ABOUT THE PROJECT -->
## About The Project

This project is inspired by Fannie Mae Single Family Loan Performance to understand more about how professional credit risk modelling can be built and expanded. Originated
from the firm's R code to conduct data analysis on the loan performance, my project is built to convert these codes to Python with application of Neural Network to predict
the probability of default for each user id. I believe this project can be a guided start for others who are familiar with Python more to expand the application and tune the
features to their wishes. Additionally, this project is designed to understand how neural network can perform time-series regression data for unrealized pattern for more complex
data points.


<!-- GETTING STARTED -->
## Getting Started

1. First, follow the [instructions](https://capitalmarkets.fanniemae.com/media/9066/display) and ignore the section for HARP data but feel free to explore as this might pique your interest
2. After having the data downloaded, you can change the original folder path in my PD.py to your location of the file in your local computer to perform the ETL process.
3. The PD.py should be the file being run first in order to generate the features needed for the neural network part
4. Finally, read the ML.py part and fine-tune the neural network or updated the algorithms in whatever you feel the best because there's no limitation at all.

### Dependencies

Here, list all libraries, packages and other dependencies that need to be installed to run your project. Include library versions and how they should be installed if a special requirement is needed.

For example, this is how you would list them:
* Pandas 2.2.2
  ```sh
  conda install -c conda-forge pandas
  ```
* pdfplumber 0.11.7
  ```sh
  conda install -c conda-forge pdfplumber
  ```
* glob 0.7
  ```sh
  conda install -c conda-forge glob
  ```
* numpy 1.26.4
  ```sh
  conda install -c conda-forge numpy
  ```
  
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/tobibui1904/Fannie-Mae-Credit-Risk-Modelling.git
   ```
2. Setup (and activate) your environment
  ```sh
  conda env create -f requirements.yml
  ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- Authors -->
## Authors

Tobi Bui - ntbui@ncsu.edu
