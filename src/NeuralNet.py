import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNet:
    def __init__(self, layers, activation="relu", output_activation="linear", lr=0.001, momentum=0.9, validation_split=0.2,  l2=0, dropout_rate=0):
        """
        Parameters:
        - layers: List defining the number of neurons in each layer, including input and output layers.
        - activation: Activation function for the hidden layers. Options: "sigmoid", "relu", "leaky_relu", "linear", and "tanh".
        - output_activation: Activation function for the output layer. Options: "linear", "relu", and "leaky_relu".
        - lr: Learning rate used for weight updates during training.
        - momentum: Momentum factor for accelerating weight updates and smoothing convergence.
        - validation_split: Proportion of data to reserve for validation during training.
        - l2: L2 regularization coefficient to prevent overfitting by penalizing large weights.
        - dropout_rate: Dropout rate for hidden layers to randomly deactivate neurons during training and prevent overfitting.
        """
        self.L = len(layers) 
        self.n = layers.copy() 
        self.lr = lr  
        self.momentum = momentum  
        self.fact = activation  
        self.output_fact = output_activation   
        self.validation_split = validation_split

        ## Regularizations
        self.l2 = l2  
        self.dropout_rate = dropout_rate

        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        self.w = [None] * self.L 
        for l in range(1, self.L):
            self.w[l] = np.random.normal(0, 0.2, (self.n[l], self.n[l-1]))

        self.theta =  [np.zeros(layers[l+1]) for l in range(len(layers) - 1)]
        self.d_w = [None] + [np.zeros((layers[i], layers[i-1])) for i in range(1, self.L)] 
        self.d_theta = [np.zeros(n) for n in layers[1:]] 
        self.d_w_prev = [None] + [np.zeros((layers[i], layers[i-1])) for i in range(1, self.L)]  
        self.d_theta_prev = [np.zeros(n) for n in layers[1:]]  

    def split_data(self, X, y):
        """
        Divides the data set into training and validation according to the percentage

        Returns:
        - X_train, y_train: train data
        - X_val, y_val: validation data 
        """
        if self.validation_split == 0:
            return X, y, None, None
    
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        return X_train, y_train, X_val, y_val


    def activation(self, z, output_layer = False):
        """
        Selected activation function
        
        """
        if output_layer:  
            if self.output_fact == "linear":
                return z
            elif self.output_fact == "relu":
                return np.maximum(0, z)
            elif self.output_fact == "leaky_relu":
                alpha = 0.01
                return np.where(z > 0, z, alpha * z)
            else:
                raise ValueError(f"Invalid '{self.output_fact}' activation function for the output layer. Valid functions for the output layer: “Linear”, “ReLu”, “Leaky ReLu”")
        else:
            if self.fact == "sigmoid":
                return 1 / (1 + np.exp(-z))
            elif self.fact == "relu":
                return np.maximum(0, z)
            elif self.fact == "leaky_relu":
                alpha = 0.01  
                return np.where(z > 0, z, alpha * z)
            elif self.fact == "linear":
                return z
            elif self.fact == "tanh":
                return np.tanh(z)
            else:
                raise ValueError("Activation function not recognized")
    
    def activation_derivative(self, z, output_layer = False):
        """
        Derivative of the selected activation function
        """
        if output_layer: 
            if self.output_fact == "linear":
                return np.ones_like(z)
            elif self.output_fact == "relu":
                return (z > 0).astype(int)
            elif self.output_fact == "leaky_relu":
                alpha = 0.01
                return np.where(z > 0, 1, alpha)
            else:
                raise ValueError(f"Invalid '{self.output_fact}' activation function for the output layer. Valid functions for the output layer: “Linear”, “ReLu”, “Leaky ReLu”")
        else:
            if self.fact == "sigmoid":
                a = self.activation(z)
                return a * (1 - a)
            elif self.fact == "relu":
                return  (z > 0).astype(int)
            elif self.fact == "leaky_relu":
                alpha = 0.01  
                return np.where(z > 0, 1, alpha)
            elif self.fact == "linear":
                return np.ones_like(z)
            elif self.fact == "tanh":
                return 1 - np.tanh(z)**2
            else:
                raise ValueError("Activation function not recognized")

    
    def forward_propagation(self, x):
        """
        Forward propagation: Calculates the activations of all layers.
        """
        # 1. Introduce the pattern in the input layer
        self.xi[0] = x  
        
        # 2. Iterate through hidden and output layers
        for l in range(1, self.L):
            #print(f"Forma de self.w[{l}]: {self.w[l].shape}")
            #print(f"Forma de self.xi[{l-1}]: {self.xi[l-1].shape}")

            # Calculate h^(ℓ)_i: 
            h = self.w[l] @ self.xi[l-1] - self.theta[l-1]
            #print(f"Layer {l}: h = {h}")
            
            # Calculate the activations and in case of dropout apply the coefficient
            activations = self.activation(h, output_layer=(l == self.L - 1))
            if l != self.L - 1 and self.dropout_rate > 0:  
                dropout_mask = (np.random.rand(*activations.shape) > self.dropout_rate).astype(float)
                activations *= dropout_mask  # Apagar neuronas
                activations /= (1 - self.dropout_rate)

            self.xi[l] = activations

        return self.xi[-1]
    
    def backward_propagation(self, x, y):
        """
        Backpropagation: Calculates gradients and adjusts weights and thresholds.
        """
        # The prediction corresponding to the input pattern
        o = self.forward_propagation(x)
        
        # Delta value in the output layer
        delta = [(o - y) * self.activation_derivative(self.xi[-1])]
        
        # Back propagate them to the rest of the network
        for l in range(self.L - 2, 0, -1):  
            z = self.w[l + 1].T @ delta[0]  
            delta.insert(0, z * self.activation_derivative(self.xi[l]))
        
        # Update of weights and threshold
        for l in range(1, self.L):

            # Update the array of matrices for the changes of the weights ¡¡Apply l2 regularization
            l2_penalty = self.l2 * self.w[l]
            self.d_w[l] = -self.lr * (delta[l-1][:, None] @ self.xi[l-1][None, :] + l2_penalty) + self.momentum * self.d_w_prev[l]
            # Update weights
            self.w[l] -= self.d_w[l]  # Actualizar pesos
            
            # Update the array of arrays for the changes of the thresholds 
            self.d_theta[l-1] = self.lr * delta[l-1] + self.momentum * self.d_theta_prev[l-1]

            # Upadate thresholds
            self.theta[l-1] -= self.d_theta[l-1] 
            
            # Save previous updates
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l-1] = self.d_theta[l-1]



    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100):
        """
        Train the neural network using backpropagation
        """

        # Split the data
        X_train, y_train, X_val, y_val = self.split_data(X, y)


        self.training_errors = []
        self.validation_errors = []

        # Check that the training start is correct
        print("Initial predictions (before training):")
        for i in range(min(5, len(X_train))):  
            pred = self.forward_propagation(X_train[i])
            print(f"Input: {X_train[i]}, Initial prediction: {pred}, Actual: {y_train[i]}")
            
        for epoch in range(epochs):

            # Choose a random pattern
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            for i in indices:
                #y_pred = self.forward_propagation(X_train[i])
                #loss = 0.5 * np.sum((y_pred - y_train[i])**2) 
                self.backward_propagation(X_train[i], y_train[i])
            
            # Calculate errors after each epoch
            train_error = np.mean([
                0.5 * np.sum((self.forward_propagation(x) - y)**2) for x, y in zip(X_train, y_train)
            ])
            self.training_errors.append(train_error)
            
            if X_val is not None and y_val is not None:
                val_error = np.mean([
                    0.5 * np.sum((self.forward_propagation(x) - y)**2) for x, y in zip(X_val, y_val)
                ])
                self.validation_errors.append(val_error)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Error: {train_error} - Val Error: {val_error if X_val is not None else 'N/A'}")
            #print(f'gradiante: {self.d_w[:25]} & {self.theta[:25]}' )

            # Select examples to be able to follow the predictions made
            if epoch % 100 == 0: 
                print("\nExamples of predictions (X_train):")
                for i in range(min(5, len(X_train))):  
                    pred = self.forward_propagation(X_train[i])
                    print(f"Predicción: {pred}, Real: {y_train[i]}")
    
    def predict(self, X):
        """
        Perform predictions
        """
        return np.array([self.forward_propagation(x) for x in X])
    
    def loss_epochs(self):
        """
        Returns the evolution of training and validation error over the epochs
        """
        if not hasattr(self, 'training_errors') or not hasattr(self, 'validation_errors'):
            raise ValueError("ERROR!! train the model")
        
        return np.array(list(zip(self.training_errors, self.validation_errors)))
    

    def cross_validate(self, X, y, k=5, epochs=100):
        """
        Perform k-fold cross-validation to evaluate the model.

        Parameters:
        - X: feature matrix.
        - y: label vector.
        - k: number of folds for cross-validation.
        - epochs: number of epochs to train the model on each fold.

        Returns:
        - Dictionary with average metrics and standard deviation (MSE, MAE, MAPE).
        """
        # Create the folds
        fold_size = len(X) // k
        indices = np.arange(len(X))
        np.random.shuffle(indices)  
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        mse_scores = []
        mae_scores = []
        mape_scores = []

        for fold_idx in range(k):
            print(f"\n=== Fold {fold_idx + 1}/{k} ===")

            # Create training and validation sets
            val_indices = folds[fold_idx]
            train_indices = np.setdiff1d(indices, val_indices)

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            self.__init__(layers=self.n, activation=self.fact, output_activation= self.output_fact, lr=self.lr, momentum=self.momentum, l2=self.l2, dropout_rate=self.dropout_rate )

            # Train the model
            self.fit(X_train, y_train, X_val, y_val, epochs=epochs)

            # Evaluate with the validation set
            y_pred = self.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")

            mse_scores.append(mse)
            mae_scores.append(mae)
            mape_scores.append(mape)

        results = {
            "MSE_mean": np.mean(mse_scores),
            "MSE_std": np.std(mse_scores),
            "MAE_mean": np.mean(mae_scores),
            "MAE_std": np.std(mae_scores),
            "MAPE_mean": np.mean(mape_scores),
            "MAPE_std": np.std(mape_scores),
        }

        print("\n=== Cross-Validation Results ===")
        print("Metric\t\tMean\t\tStd Dev")
        print(f"MSE\t\t{results['MSE_mean']:.4f}\t\t{results['MSE_std']:.4f}")
        print(f"MAE\t\t{results['MAE_mean']:.4f}\t\t{results['MAE_std']:.4f}")
        print(f"MAPE\t\t{results['MAPE_mean']:.4f}%\t\t{results['MAPE_std']:.4f}%")

        return results

