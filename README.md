## Evaluating different deep learning models for efficient extraction of Raman signals from CARS spectra

Paper title: "Evaluating different deep learning models for efficient extraction of Raman signals from CARS spectra" 

By [Rajendhar junjuri](https://scholar.google.co.in/citations?user=BRu_wuAAAAAJ&hl=en)\, [Ali Saghi](https://scholar.google.co.in/citations?view_op=list_works&hl=en&hl=en&user=GcWhnFcAAAAJ),  [Lasse Lensu](https://scholar.google.co.in/citations?user=dk2Ezl0AAAAJ&hl=en&oi=ao), and [Erik M. Vartiainen](https://scholar.google.co.in/citations?user=zbxe2qYAAAAJ&hl=en&oi=ao) 

These four models are different from each other viz, convolutional neural network (CNN), Long short-term memory (LSTM) neural network, very deep convolutional autoencoders (VECTOR), and Bi-directional LSTM (Bi – LSTM) neural network. 

## The article can be accessed here
DOI: https://doi.org/10.1039/D3CP01618H

## Citation
Junjuri, R., Saghi, A., Lensu, L., & Vartiainen, E. M. (2023). Evaluating different deep learning models for efficient extraction of Raman signals from CARS spectra. Physical Chemistry Chemical Physics, 25(24), 16340-16353.

## More related articles can be found here
https://scholar.google.co.in/citations?hl=en&user=BRu_wuAAAAAJ&view_op=list_works&sortby=pubdate

## About Synthetic test data and code
The details of the 300 synthetic test spectra

"y_test_300_merge_spectra3.npy"---> referes to the true Raman signal and data can be found here https://github.com/Junjuri/LUT/blob/main/y_test_300_merge_spectra3.npy

"x_test_300_merge_spectra3.npy"---> referes to the input CARS data and data can be found here https://github.com/Junjuri/LUT/blob/main/x_test_300_merge_spectra3.npy

Testing can be done by using the following program.
'RSS_Advances_CNN_prediction_on_test_data.py'. It can be found here https://github.com/Junjuri/LUT/blob/main/RSS_Advances_CNN_prediction_on_test_data.py

## About the experimental CARS test data
The experimental CARS test data set used in this investigation can only be provided upon request and can contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843?auxfun=&lang=en_GB) 

## About the training codes

1. CNN model trainin can be done by using this program 'RSS_Advances_CNN_to_train_with_different_NRBs.py' Please see here     https://github.com/Junjuri/LUT/blob/main/RSS_Advances_CNN_to_train_with_different_NRBs.py

2. Bi-LSTM model training can be done by using this program 'bi-LSTM_train.py'. 

3. LSTM model training can be done by using this program 'LSTM_train.py'. 

4. Vector model training can be done by using this program


## About the trained model weights

"CNN_model_weights can be found here https://github.com/Junjuri/LUT/blob/main/Polynomial_NRB_model_weights.h5 with a name 'Polynomial_NRB_model_weights.h5' 

"LSTM_model_weights.h5" --->referes weights of the model trained with LSTM.

"VECTOR_model_weights.h5" --->referes weights of the model trained with VECTOR can be accseed via request at ali.saghi.2015@gmail.com or rajendhar.j2008@gmail.com

"Bi_LSTM_model_weights.h5" --->referes weights of the model trained with Bi-LSTM

## Getting Started and Requirements 
You can use Python (TensorFlow 2.7.0) to test the pre-trained network. We have tested it in Spyder.
