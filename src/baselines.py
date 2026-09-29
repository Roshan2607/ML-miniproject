import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

class SVRBaseline:
    """
    Support Vector Regression (SVR) baseline model.
    As per the paper: "we have a support vector multi-regressor for training each input text matrix 
    and an array of spectrogram output (length 1025)."
    """
    def __init__(self, kernel='poly', use_incremental=False):
        self.use_incremental = use_incremental
        
        if use_incremental:
            # The paper attempted online passive-aggressive algorithm to speed up training
            base_estimator = PassiveAggressiveRegressor(C=1.0, random_state=42)
        else:
            # Regular polynomial kernel SVR which got better MOS
            base_estimator = SVR(kernel=kernel)
            
        self.model = MultiOutputRegressor(base_estimator)

    def fit(self, X_train, Y_train):
        """
        X_train: 2D array of shape (num_samples, max_seq_len)
        Y_train: 2D array of flattened spectrograms (num_samples, target_len * 1025)
        """
        print(f"Training SVR model (Incremental: {self.use_incremental}). This may take a long time...")
        self.model.fit(X_train, Y_train)
        print("Training complete.")

    def predict(self, X_test, target_shape):
        """
        Returns predictions reshaped back to 3D matrices for post-processing.
        target_shape should be (target_len, 1025)
        """
        predictions_1d = self.model.predict(X_test)
        
        num_samples = predictions_1d.shape[0]
        # Reshape the prediction vectors back to matrices (as mentioned in the paper)
        predictions_3d = predictions_1d.reshape(num_samples, target_shape[0], target_shape[1])
        return predictions_3d
