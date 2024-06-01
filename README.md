# Deep Learning Coursework

## Time Series Prediction: Air Quality Index (Assignment 1 Part 3)

### Data Preparation
1. **Loading Data:**
   - The Air Quality UCI dataset is loaded into a Pandas DataFrame.
   - The first few rows are displayed to understand the structure of the dataset.

2. **Data Cleaning:**
   - Dropped the last two columns containing NaN values.
   - Removed any remaining rows with missing values.
   - Displayed the column data types and main statistics to understand the data distribution and identify potential issues.

3. **Feature Selection:**
   - Dropped the 'Date' and 'Time' columns as they are not useful for correlation and prediction purposes.
   - Visualized correlations using pair plots and a heatmap to identify relationships between features.

4. **Normalization:**
   - Standardized the dataset to have a mean of 0 and a standard deviation of 1 using `StandardScaler`.
   - Separated features (X) and the target variable (y).

### Data Transformation
1. **Tensor Conversion:**
   - Converted the features and target variable into PyTorch tensors.
   
2. **Sequence Preparation:**
   - Transformed the dataset into sequences to prepare it for time series analysis using RNN.

3. **Data Splitting:**
   - Split the data into training, validation, and test sets using an 80-20 split for training/validation and then a 50-50 split of the remaining data for validation/test.

4. **DataLoader Creation:**
   - Created PyTorch DataLoader objects for training, validation, and test sets to facilitate batch processing.

### Model Building
1. **RNN Model Definition:**
   - Defined an RNN model using PyTorch with:
     - An RNN layer
     - An LSTM layer
     - A fully connected layer for the final output
   - Specified input size, hidden size, LSTM hidden size, output size, and number of layers.

2. **Training Setup:**
   - Defined the loss function (`MSELoss`) and optimizer (`Adam`).
   - Set up parameters such as learning rate and number of epochs.

### Model Training
1. **Training Loop:**
   - Implemented a training loop to train the model over a specified number of epochs.
   - Calculated and recorded training and validation losses and R2 scores at each epoch.
   - Visualized the training and validation loss and R2 scores over epochs.

#### Training Results:
   - Epoch 1/10, Train Loss: 0.6527, Val Loss: 0.2032, Train R2: -0.1404, Val R2: 0.8097
   - Epoch 2/10, Train Loss: 0.1052, Val Loss: 0.0586, Train R2: -0.6784, Val R2: 0.9451
   - Epoch 3/10, Train Loss: 0.0798, Val Loss: 0.0071, Train R2: -0.8755, Val R2: 0.9934
   - Epoch 4/10, Train Loss: 0.0601, Val Loss: 0.0147, Train R2: -0.9466, Val R2: 0.9862
   - Epoch 5/10, Train Loss: 0.0753, Val Loss: 0.0035, Train R2: -0.9159, Val R2: 0.9967
   - Epoch 6/10, Train Loss: 0.0660, Val Loss: 0.0052, Train R2: -0.9605, Val R2: 0.9952
   - Epoch 7/10, Train Loss: 0.0744, Val Loss: 0.0077, Train R2: -0.9344, Val R2: 0.9928
   - Epoch 8/10, Train Loss: 0.0677, Val Loss: 0.0082, Train R2: -0.9505, Val R2: 0.9923
   - Epoch 9/10, Train Loss: 0.0679, Val Loss: 0.0077, Train R2: -0.9418, Val R2: 0.9928
   - Epoch 10/10, Train Loss: 0.0697, Val Loss: 0.0040, Train R2: -0.9222, Val R2: 0.9962

### Model Evaluation
1. **Test Evaluation:**
   - Evaluated the model on the test set to calculate the final loss, mean absolute error (MAE), and root mean squared error (RMSE).
   - Calculated the R2 score for the test set.
   - Test R2 Score: 0.996259

2. **Model Saving:**
   - Saved the trained model's state dictionary for future use.

### Optimized Model
1. **Optimized RNN Model Definition:**
   - Added dropout and L2 regularization (weight decay) to the RNN model for better generalization and to prevent overfitting.

2. **Training and Evaluation:**
   - Repeated the training and evaluation process with the optimized model.
   - Compared performance metrics (loss, MAE, RMSE, R2 score) to the initial model.


## Anomaly Detection (Assignment 2 Part 2)

**Dataset: Hard Drive Test Data**
- **Source:** [Kaggle](https://www.kaggle.com/datasets/backblaze/hard-drive-test-data?resource=download)
- **Description:** This dataset contains daily health check-ups of hard drives, detailing S.M.A.R.T. stats, model info, capacity, and failure status. The goal is to predict potential failures to prevent data loss and ensure system reliability.

**Models:**

1. **Simple Autoencoder Model:**
   - **Encoder:** Compresses data through linear layers reducing dimensions from input size to 128, 64, 12, and finally 3 units, using ReLU activation.
   - **Decoder:** Reconstructs data inversely, expanding dimensions from 3 units back to the original size with a Sigmoid activation in the final layer.
   
2. **LSTM Autoencoder Model:**
   - **Configuration:** Uses 128 hidden units, 1 layer, and batch-first processing.
   - **Encoder:** Compresses sequences via linear layers reducing dimensions to a 3-unit bottleneck with ReLU activations.
   - **Decoder:** Reconstructs sequences from the bottleneck back to original dimensions with ReLU activations and a final Sigmoid layer.
   
3. **Compact Deep AutoEncoder:**
   - **Encoder:** Reduces dimensions through layers down to a 3-unit bottleneck with ReLU activation.
   - **Decoder:** Expands dimensions back to original size, using ReLU for intermediate layers and Sigmoid for the final layer.

**Evaluation Metrics:**
- **R2 Score, Reconstruction Error, and Losses:** Used to assess the models' performance in detecting anomalies by measuring the accuracy of the reconstructed data compared to the original data.

### Review Polarity (Assignment 2 Part 4)

**Preprocessing and Visualization:**
- **Visualizations:** Pie chart for positive/negative samples, distribution of review lengths, and box plot for review length vs polarity.
- **Text Preprocessing:** Lowercasing, removing punctuation, tokenization, and stopword removal.

**Transformer Architecture:**
- **Embedding Layer:** Converts tokens to dense vectors.
- **Positional Embedding:** Adds sequence order information.
- **Transformer Encoder Layers:** Stacked self-attention and feed-forward layers.
- **Classification Layer:** Linear classifier for final output.

**Training:**
- **Method:** Cross-entropy loss, Adam optimizer.
- **Performance:** 
  - **Testing Loss:** 0.2981
  - **Testing Accuracy:** 88.02%
  - **Precision:** 0.8923
  - **Recall:** 0.9215
  - **F1 Score:** 0.9067
  - **ROC Curve:** Visualizes performance.

**Regularization Techniques:**
- **Dropout, L2 Regularization, and Early Stopping:**
  - **Dropout:** Slight decrease in accuracy.
  - **L2 Regularization:** Over-penalized parameters, lower accuracy.
  - **Early Stopping:** Lower accuracy, potential for further training improvement.

**Model Performance:**
- **Base Model:** Highest validation accuracy of 90.34%, stable improvement over epochs.
- **Dropout:** Slight drop in performance with validation accuracy of 89.36%.
- **L2 Regularization:** Validation accuracy reduced to 88.32%, indicating over-penalization.
- **Early Stopping:** Validation accuracy of 88.03%, suggesting potential for additional training.
