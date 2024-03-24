# Prediction of Patient Admission Probability Model

This model aims to address the challenge of handling data with variable feature lengths. The fundamental concept is to convert the variable-length feature vectors into fixed-length representations through the use of embedding and feature extraction techniques, which are then fed into a classifier for classification purposes.

In this model, Long Short-Term Memory (LSTM) networks are employed for feature extraction, leveraging their ability to effectively capture and learn from time series data. Following data preprocessing, the time-related features are extracted and input into the LSTM for training. The resulting trained feature vectors are subsequently concatenated with the remaining feature vectors and provided as input to the classifier for further training.

The time-related features are categorized into temporal and spatial components. To optimally fit both aspects, this model utilizes the product method, ensuring that both the temporal and spatial features are preserved to the greatest extent possible.

## Flow Chart

![](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403241456453.png)

## Results

### Data Preprocessing

![](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403240058645.png)

see `patients.csv` file

### LSTM + XGBM

![image-20240324010220549](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403240102582.png)

### LSTM + LightGBM

![image-20240324010250082](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403240102119.png)

### LSTM + Random Forest

![image-20240324010347077](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403240103107.png)

### LSTM

![image-20240324010113629](https://raw.githubusercontent.com/WYX-K/pic_bed/main/img/202403240101668.png)

## Platform

- Windows 11
- Python 3.12
- pytorch 2.2.1
- cuda 11.8
- cudnn 8.2.4

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

see `all_in_one_test.ipynb` file for the usage of the model.
