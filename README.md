# Four-models-comparison

These four models are different from each other viz, convolutional neural network (CNN), Long short-term memory (LSTM) neural network, very deep convolutional autoencoders (VECTOR), and Bi-directional LSTM (Bi – LSTM) neural network. 

Paper title: "Evaluating different deep learning models for efficient extraction of Raman signals from CARS spectra"

By [Rajendhar junjuri](https://scholar.google.co.in/citations?user=BRu_wuAAAAAJ&hl=en)\, [Ali Saghi](https://scholar.google.co.in/citations?view_op=list_works&hl=en&hl=en&user=GcWhnFcAAAAJ),  [Lasse Lensu](https://scholar.google.co.in/citations?user=dk2Ezl0AAAAJ&hl=en&oi=ao), and [Erik M. Vartiainen](https://scholar.google.co.in/citations?user=zbxe2qYAAAAJ&hl=en&oi=ao) 

## About Synthetic test data
These are 300 synthetic test spectra can be found here 

"y_test_300_merge_spectra3.npy"---> referes to the true Raman signal

"x_test_300_merge_spectra3.npy"---> referes to the input CARS data

## About the experimental CARS test data
The experimental CARS test data set used in this investigation can only be provided upon request and can contact [Erik M. Vartiainen](https://research.lut.fi/converis/portal/detail/Person/56843?auxfun=&lang=en_GB) 

## About the training codes

The model architecture is directly adapted from the SpecNet paper (See https://github.com/Valensicv/SpecNet for the full code of the neural network model)
Here three different NRBs are evaluated. 

It can be accessed from the following program.
RSS_Advances_CNN_to_train_with_different_NRBs.py

Testing can be done by using the following program.
RSS_Advances_CNN_prediction_on_test_data.py

## About the trained model weights

"CNN_model_weights can be found here https://github.com/Junjuri/LUT/blob/main/Polynomial_NRB_model_weights.h5 with a name 'Polynomial_NRB_model_weights.h5' 

"LSTM_model_weights.h5" --->referes weights of the model trained with LSTM.

"VECTOR_model_weights.h5" --->referes weights of the model trained with VECTOR

"Bi_LSTM_model_weights.h5" --->referes weights of the model trained with Bi-LSTM can be accseed vai request at rajendhar.j2008@gmail.com

## Getting Started and Requirements 
You can use Python (TensorFlow 2.7.0) to test the pre-trained network. We have tested it in Spyder.

## Citation
Formats to cite our paper will be updated soon
