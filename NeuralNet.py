import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuralNet:
    def __init__(self, layers, activation="relu", lr=0.001, momentum=0.9, validation_split=0.2,  l2=0, dropout_rate=0):
        """
        Constructor de la clase NeuralNet.
        
        Parámetros:
        - layers: lista con el número de neuronas por capa (incluyendo entrada y salida).
        - activation: función de activación ("sigmoid", "relu", "linear", "tanh").
        - lr: tasa de aprendizaje.
        - momentum: factor de momento para el ajuste de pesos.
        """
        self.L = len(layers) 
        self.n = layers.copy() 
        self.lr = lr  
        self.momentum = momentum  
        self.fact = activation  
        self.validation_split = validation_split

        ## Regularizations
        self.l2 = l2  
        self.dropout_rate = dropout_rate


        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(layers[lay]))

        # self.w = [None] * self.L  # Lista para los pesos
        # for l in range(1, self.L):
        #     self.w[l] = np.random.randn(self.n[l], self.n[l-1]) * np.sqrt(2 / self.n[l-1])


        self.w = [None] * self.L  # Lista para los pesos
        for l in range(1, self.L):
            self.w[l] = np.random.normal(0, 0.1, (self.n[l], self.n[l-1]))

        # self.w = [None] * self.L  # Lista para los pesos
        # for l in range(1, self.L):
        #     self.w[l] = np.random.randn(self.n[l], self.n[l-1]) *0.1

        # self.w = []
        # self.w.append(np.zeros((1, 1)))
        # for lay in range(1, self.L):
        #     self.w.append(np.zeros((layers[lay], layers[lay - 1])))

        #self.w = [None] + [np.random.randn(layers[i], layers[i-1]) * np.sqrt(1 / layers[i-1]) for i in range(1, self.L)]

        self.theta = [np.random.randn(self.n[l]) * 0.01 for l in range(1, self.L)]  # Umbrales (sin incluir capa de entrada)
        self.d_w = [None] + [np.zeros((layers[i], layers[i-1])) for i in range(1, self.L)]  # Gradientes de pesos
        self.d_theta = [np.zeros(n) for n in layers[1:]]  # Gradientes de umbrales
        self.d_w_prev = [None] + [np.zeros((layers[i], layers[i-1])) for i in range(1, self.L)]  # Pesos anteriores
        self.d_theta_prev = [np.zeros(n) for n in layers[1:]]  # Umbrales anteriores

    def split_data(self, X, y):
        """
        Divide el conjunto de datos en entrenamiento y validación según el porcentaje especificado.

        Parameters:
        - X: matriz de características.
        - y: vector de etiquetas.

        Returns:
        - X_train, y_train: datos de entrenamiento.
        - X_val, y_val: datos de validación (si validation_split > 0).
        """
        if self.validation_split == 0:
            return X, y, None, None

        # Barajar los datos
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        # Dividir en entrenamiento y validación
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        return X_train, y_train, X_val, y_val



    def activation(self, z, output_layer = False):
        """
        Función de activación seleccionada.
        
        Parámetros:
        - z: entrada a la función de activación.
        
        Retorna:
        - Salida de la función de activación.
        """
        if output_layer:
            return z
        elif self.fact == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.fact == "relu":
            return np.maximum(0, z)
        elif self.fact == "linear":
            return z
        elif self.fact == "tanh":
            return np.tanh(z)
        else:
            raise ValueError("Función de activación no reconocida.")
    
    def activation_derivative(self, z):
        """
        Derivada de la función de activación seleccionada.
        
        Parámetros:
        - z: entrada a la función de activación.
        
        Retorna:
        - Derivada de la función de activación.
        """
        if self.fact == "sigmoid":
            a = self.activation(z)
            return a * (1 - a)
        elif self.fact == "relu":
            return  (z > 0).astype(int)
        elif self.fact == "linear":
            return np.ones_like(z)
        elif self.fact == "tanh":
            return 1 - np.tanh(z)**2
        else:
            raise ValueError("Función de activación no reconocida.")

    
    def forward_propagation(self, x):
        """
        Propagación hacia adelante: Calcula las activaciones de todas las capas.
        """
        # 1. Introducir el patrón en la capa de entrada
        #self.xi = [None] * self.L
        self.xi[0] = x  # ξ^(1) = x
        
        # 2. Iterar por las capas ocultas y de salida
        for l in range(1, self.L):
            #print(f"Forma de self.w[{l}]: {self.w[l].shape}")
            #print(f"Forma de self.xi[{l-1}]: {self.xi[l-1].shape}")


            # Calcular h^(ℓ)_i: h = W * ξ + b (o W*x - theta)
            h = self.w[l] @ self.xi[l-1] - self.theta[l-1]
            #print(f"Layer {l}: h = {h}")
            
            # Aplicar función de activación: ξ^(ℓ) = g(h^(ℓ)) with dropout
            drop = self.activation(h, output_layer=(l == self.L - 1))
            if l != self.L - 1 and self.dropout_rate > 0:  # Aplicar dropout en capas ocultas
                dropout_mask = (np.random.rand(*drop.shape) > self.dropout_rate).astype(float)
                drop *= dropout_mask  # Apagar neuronas
                drop /= (1 - self.dropout_rate)

            self.xi[l] = drop

        
        # 3. Retornar la salida de la red: ξ^(L) (última capa)
        return self.xi[-1]
    
    def backward_propagation(self, x, y):
        """
        Retropropagación: Calcula los gradientes y ajusta pesos y umbrales.
        """
        # Paso hacia adelante para obtener salida
        o = self.forward_propagation(x)
        
        # Inicializar delta en la capa de salida
        delta = [(o - y) * self.activation_derivative(self.xi[-1])]
        
        # Calcular deltas para capas ocultas
        for l in range(self.L - 2, 0, -1):  # Desde capa L-1 hacia 1
            z = self.w[l + 1].T @ delta[0]  # Error propagado hacia atrás
            delta.insert(0, z * self.activation_derivative(self.xi[l]))
        
        # Definir un valor para el clipping de los gradientes
        grad_clip_value = 5  # Ajusta según sea necesario
        
        # Actualizar pesos y umbrales
        for l in range(1, self.L):
            # Gradiente de pesos (w)
            l2_penalty = self.l2 * self.w[l]
            self.d_w[l] = -self.lr * (delta[l-1][:, None] @ self.xi[l-1][None, :] + l2_penalty) + self.momentum * self.d_w_prev[l]
            
            # Clipping de gradientes para evitar valores extremos
            #self.d_w[l] = np.clip(self.d_w[l], -grad_clip_value, grad_clip_value)

            # Actualizar pesos
            self.w[l] -= self.d_w[l]  # Actualizar pesos
            
            # Gradiente de umbrales (θ)
            self.d_theta[l-1] = self.lr * delta[l-1] + self.momentum * self.d_theta_prev[l-1]
            
            # Clipping de gradientes para los umbrales
            #self.d_theta[l-1] = np.clip(self.d_theta[l-1], -grad_clip_value, grad_clip_value)

            # Actualizar umbrales
            self.theta[l-1] -= self.d_theta[l-1]  # Actualizar umbrales
            
            # Guardar las actualizaciones previas (para el momentum)
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l-1] = self.d_theta[l-1]



    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100):
        """
        Entrena la red neuronal usando backpropagation.
        """

        X_train, y_train, X_val, y_val = self.split_data(X, y)


        self.training_errors = []
        self.validation_errors = []

        print("Predicciones iniciales (antes del entrenamiento):")
        for i in range(min(5, len(X_train))):  # Mostrar un máximo de 5 ejemplos
            pred = self.forward_propagation(X_train[i])
            print(f"Entrada: {X_train[i]}, Predicción inicial: {pred}, Real: {y_train[i]}")
            
        for epoch in range(epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            # Entrenamiento por patrones
            for i in indices:
                y_pred = self.forward_propagation(X_train[i])
                loss = 0.5 * np.sum((y_pred - y_train[i])**2)  # Error del ejemplo
                self.backward_propagation(X_train[i], y_train[i])
            
            # Calcular errores después de cada época
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
            # Seleccionar algunos ejemplos para inspección

            if epoch % 100 == 0:  # Imprime cada 10 épocas
                print("\nEjemplos de predicciones (X_train):")
                for i in range(min(5, len(X_train))):  # Máximo 5 ejemplos
                    pred = self.forward_propagation(X_train[i])
                    print(f"Predicción: {pred}, Real: {y_train[i]}")
    
    def predict(self, X):
        """
        Predice las salidas para un conjunto de entradas.
        """
        return np.array([self.forward_propagation(x) for x in X])
    
    def loss_epochs(self):
        """
        Retorna la evolución del error de entrenamiento y validación a lo largo de las épocas.

        Retorna:
        - Un array con dos columnas: errores de entrenamiento y validación por época.
        """
        if not hasattr(self, 'training_errors') or not hasattr(self, 'validation_errors'):
            raise ValueError("Debes entrenar el modelo con `fit` antes de calcular `loss_epochs`.")
        
        # Crear matriz con errores ya calculados durante fit
        return np.array(list(zip(self.training_errors, self.validation_errors)))
    

    def cross_validate(self, X, y, k=5, epochs=100):
        """
        Realiza k-fold cross-validation para evaluar el modelo.

        Parámetros:
        - X: matriz de características.
        - y: vector de etiquetas.
        - k: número de pliegues para cross-validation.
        - epochs: número de épocas para entrenar el modelo en cada pliegue.

        Retorna:
        - results: diccionario con métricas promedio y desviación estándar (MSE, MAE, MAPE).
        """
        # Crear índices para k-fold
        fold_size = len(X) // k
        indices = np.arange(len(X))
        np.random.shuffle(indices)  # Barajar para aleatoriedad
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

        mse_scores = []
        mae_scores = []
        mape_scores = []

        for fold_idx in range(k):
            print(f"\n=== Fold {fold_idx + 1}/{k} ===")
            # Crear conjuntos de entrenamiento y validación
            val_indices = folds[fold_idx]
            train_indices = np.setdiff1d(indices, val_indices)

            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # Reinicializar el modelo antes de cada entrenamiento
            self.__init__(layers=self.n, activation=self.fact, lr=self.lr, momentum=self.momentum)

            # Entrenar el modelo
            self.fit(X_train, y_train, X_val, y_val, epochs=epochs)

            # Evaluar el modelo en el conjunto de validación
            y_pred = self.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%")

            mse_scores.append(mse)
            mae_scores.append(mae)
            mape_scores.append(mape)

        # Calcular promedio y desviación estándar de cada métrica
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

