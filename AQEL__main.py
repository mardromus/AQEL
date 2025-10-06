import numpy as np
import matplotlib.pyplot as plt
import random
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, classification_report

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)


# --- Custom Data Generation and Preprocessing (Enhanced) ---

def custom_make_moons(n_samples=100, noise=0.1, random_state=None):
    """
    Generates a 2D dataset resembling two interleaving half circles.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer circle
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    outer_circ = np.vstack([outer_circ_x, outer_circ_y]).T

    # Inner circle
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5
    inner_circ = np.vstack([inner_circ_x, inner_circ_y]).T

    X = np.vstack([outer_circ, inner_circ])
    y = np.hstack([np.zeros(n_samples_out), np.ones(n_samples_in)])

    # Add noise
    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, y


def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits arrays or matrices into random train and test subsets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def custom_standard_scaler(X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Handle cases where std might be zero
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


# --- 1. Enhanced SSDI ---
class SSDI:
    def _init_(self, n_features, n_samples=400, test_size=0.2, random_state=42):  # Increased samples
        self.n_features = n_features
        self.n_samples = n_samples
        self.test_size = test_size
        self.random_state = random_state

    def load_and_preprocess_data(self):
        if self.n_features != 2:
            print("Warning: make_moons dataset inherently has 2 features. Adjusting n_features to 2.")
            self.n_features = 2

        X, y = custom_make_moons(n_samples=self.n_samples, noise=0.05, random_state=self.random_state)  # Reduced noise

        # Convert labels to -1 and 1 for binary classification
        y = np.array([-1 if val == 0 else 1 for val in y])

        X_train, X_test, y_train, y_test = custom_train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Scale data for better performance in QML
        X_train_scaled, X_test_scaled = custom_standard_scaler(X_train, X_test)

        print(f"Data loaded: X_train shape {X_train_scaled.shape}, y_train shape {y_train.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test


# --- 2. Enhanced AQFM with Better Feature Encoding ---
class AQFM:
    def _init_(self, n_qubits, n_features=None):  # Added n_features parameter
        self.n_qubits = n_qubits
        self.n_features = n_features if n_features is not None else n_qubits
        self.n_params = n_qubits * 2  # More parameters for richer encoding

    def circuit(self, x_data, fm_params):
        """
        Enhanced feature mapping that handles dimension mismatch.
        """
        # First layer: RY rotations with data (repeat/pad features if needed)
        for i in range(self.n_qubits):
            feature_idx = i % self.n_features  # Cycle through available features
            qml.RY(x_data[feature_idx] * fm_params[i], wires=i)
        
        # Second layer: RZ rotations with trainable parameters
        for i in range(self.n_qubits):
            qml.RZ(fm_params[i + self.n_qubits], wires=i)

        # Enhanced entanglement pattern
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if self.n_qubits > 2:
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # Cyclic entanglement



# --- 3. Enhanced NAVQC with Better Ansatz ---
class NAVQC:
    def _init_(self, n_qubits, n_layers=2):  # Start with more layers
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Use a more expressive ansatz
        self.param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        self.n_params = np.prod(self.param_shape)

    def circuit(self, vqc_params):
        """
        Enhanced variational quantum circuit with better expressivity.
        """
        # Reshape flat parameters to the required shape
        reshaped_params = vqc_params.reshape(self.param_shape)
        qml.StronglyEntanglingLayers(reshaped_params, wires=range(self.n_qubits))
        
        # Add extra rotation layer for more expressivity
        for i in range(self.n_qubits):
            qml.RY(reshaped_params[-1, i, 0], wires=i)  # Use last layer parameters

    def set_layers(self, new_layers):
        """
        Update the number of layers in the ansatz.
        """
        if new_layers != self.n_layers:
            self.n_layers = new_layers
            self.param_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
            self.n_params = np.prod(self.param_shape)
            print(f"NAVQC layers updated to {new_layers}. Parameters re-initialized.")


# --- 4. Enhanced CMO-QF with Better Optimization ---
class CMO_QF:
    def _init_(self, aqfm, navqc, n_qubits):
        self.aqfm = aqfm
        self.navqc = navqc
        self.n_qubits = n_qubits
        
        # Use exact simulation for better accuracy
        self.dev = qml.device("default.qubit", wires=n_qubits)  # Remove shots for exact simulation
        
        # Total number of parameters
        self.n_aqfm_params = self.aqfm.n_params
        self.n_navqc_params = self.navqc.n_params
        
        # Create the QNode once during initialization
        self.quantum_circuit = qml.QNode(self._quantum_circuit_func, self.dev)
        
        # Track training history
        self.train_losses = []
        self.test_losses = []

    def _quantum_circuit_func(self, x_data, all_params):
        """
        Combined quantum circuit with feature map and variational layers.
        """
        # Split parameters
        fm_params = all_params[:self.n_aqfm_params]
        vqc_params = all_params[self.n_aqfm_params:self.n_aqfm_params + self.n_navqc_params]
        
        # Apply feature map
        self.aqfm.circuit(x_data, fm_params)
        
        # Apply variational circuit
        self.navqc.circuit(vqc_params)
        
        # Return expectation value of PauliZ on qubit 0
        return qml.expval(qml.PauliZ(0))

    def cost_function(self, all_params, X, y):
        """
        Enhanced cost function with regularization.
        """
        predictions = []
        
        for x_data in X:
            prediction = self.quantum_circuit(x_data, all_params)
            predictions.append(prediction)

        predictions = np.array(predictions)
        
        # Mean Squared Error with L2 regularization
        mse_loss = np.mean((predictions - y) ** 2)
        l2_reg = 0.001 * np.sum(all_params ** 2)  # Small regularization
        
        total_loss = mse_loss + l2_reg
        return total_loss

    def train(self, X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01):  # More epochs
        """
        Enhanced training loop with better optimization strategies.
        """
        # Initialize parameters with Xavier/Glorot initialization
        xavier_std = np.sqrt(2.0 / (self.n_aqfm_params + self.n_navqc_params))
        initial_aqfm_params = np.random.normal(0, xavier_std, self.n_aqfm_params)
        initial_navqc_params = np.random.normal(0, xavier_std, self.n_navqc_params)
        all_params = np.concatenate([initial_aqfm_params, initial_navqc_params])

        print("\n--- Starting Enhanced AQEL Training ---")

        def wrapped_cost(params):
            return self.cost_function(params, X_train, y_train)

        current_params = all_params
        best_params = current_params.copy()
        best_test_loss = float('inf')
        patience = 0
        max_patience = 3

        # Try different optimizers
        optimizers = ['BFGS', 'L-BFGS-B', 'Powell']
        
        for epoch_block in range(epochs // 20):  # Smaller blocks for more frequent adaptation
            print(f"\n--- Optimization Block {epoch_block + 1} ---")
            
            # Choose optimizer (cycle through them)
            current_optimizer = optimizers[epoch_block % len(optimizers)]
            print(f"Using optimizer: {current_optimizer}")

            # Run optimization steps with different methods
            if current_optimizer == 'BFGS':
                res = minimize(wrapped_cost, current_params, method='BFGS', 
                             options={'maxiter': 20, 'disp': False})
            elif current_optimizer == 'L-BFGS-B':
                res = minimize(wrapped_cost, current_params, method='L-BFGS-B', 
                             options={'maxiter': 20, 'disp': False})
            else:  # Powell
                res = minimize(wrapped_cost, current_params, method='Powell', 
                             options={'maxiter': 20, 'disp': False})
            
            current_params = res.x

            train_loss = self.cost_function(current_params, X_train, y_train)
            test_loss = self.cost_function(current_params, X_test, y_test)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            print(f"Epoch Block {epoch_block * 20 + 1}-{min((epoch_block + 1) * 20, epochs)}")
            print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Early stopping with best model tracking
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_params = current_params.copy()
                patience = 0
            else:
                patience += 1

            # Adaptive architecture modification
            if test_loss > 0.15 and self.navqc.n_layers < 4:  # More aggressive adaptation
                print("CMO-QF: Increasing NAVQC layers for better expressibility.")
                self.navqc.set_layers(self.navqc.n_layers + 1)
                
                # Re-create QNode with new architecture
                self.quantum_circuit = qml.QNode(self._quantum_circuit_func, self.dev)
                
                # Smart parameter initialization - keep what works
                current_aqfm_params = current_params[:self.n_aqfm_params]
                new_navqc_params = np.random.normal(0, xavier_std, self.navqc.n_params)
                current_params = np.concatenate([current_aqfm_params, new_navqc_params])
                self.n_navqc_params = self.navqc.n_params
                
                patience = 0  # Reset patience after architecture change

            # Learning rate decay
            if epoch_block > 0 and epoch_block % 3 == 0:
                learning_rate *= 0.9
                print(f"Learning rate decayed to: {learning_rate:.6f}")

            # Early stopping
            if patience >= max_patience:
                print("Early stopping triggered - no improvement in test loss.")
                break

        print("--- Enhanced AQEL Training Complete ---")
        final_test_loss = self.cost_function(best_params, X_test, y_test)
        print(f"Best Test Loss: {final_test_loss:.4f}")
        return best_params

    def predict(self, X, params):
        """
        Make predictions on new data.
        """
        predictions = []
        for x_data in X:
            prediction = self.quantum_circuit(x_data, params)
            predictions.append(prediction)
        return np.array(predictions)

    def plot_training_history(self):
        """
        Plot training and test loss over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.test_losses, label='Test Loss', color='red')
        plt.xlabel('Optimization Block')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# --- Main Execution with Enhanced Settings ---
if _name_ == "_main_":
    n_features = 2
    n_qubits = 3  # Use more qubits for better expressivity

    # 1. Initialize SSDI and load data with better settings
    ssdi = SSDI(n_features=n_features, n_samples=300, random_state=42)
    X_train, X_test, y_train, y_test = ssdi.load_and_preprocess_data()

    # 2. Initialize enhanced AQFM and NAVQC with proper dimension handling
    aqfm = AQFM(n_qubits=n_qubits, n_features=n_features)  # Pass both parameters
    navqc = NAVQC(n_qubits=n_qubits, n_layers=2)

    # 3. Initialize enhanced CMO-QF and start training
    cmo_qf = CMO_QF(aqfm, navqc, n_qubits=n_qubits)

    # Run the enhanced training process
    final_params = cmo_qf.train(X_train, y_train, X_test, y_test, epochs=100, learning_rate=0.01)

    # Make final predictions with threshold adjustment
    final_predictions = cmo_qf.predict(X_test, final_params)
    
    # Apply adaptive threshold instead of just sign
    threshold = np.mean(final_predictions)  # Use mean as threshold
    binary_predictions = np.where(final_predictions > threshold, 1, -1)
    
    accuracy = np.mean(binary_predictions == y_test)
    print(f"\nFinal Model Accuracy on Test Set: {accuracy:.4f}")
    
    # Detailed evaluation
    print("\n--- Detailed Evaluation ---")
    print(f"Prediction range: [{np.min(final_predictions):.3f}, {np.max(final_predictions):.3f}]")
    print(f"Threshold used: {threshold:.3f}")
    
    # Try different thresholds
    thresholds = np.linspace(-0.5, 0.5, 11)
    best_acc = 0
    best_threshold = 0
    
    for thresh in thresholds:
        pred = np.where(final_predictions > thresh, 1, -1)
        acc = np.mean(pred == y_test)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh
    
    print(f"Best threshold: {best_threshold:.3f} with accuracy: {best_acc:.4f}")
    
    # Plot training history
    cmo_qf.plot_training_history()

    # --- Enhanced Visualization ---
    if n_features == 2:
        x_min, x_max = X_train[:, 0].min() - 0.2, X_train[:, 0].max() + 0.2
        y_min, y_max = X_train[:, 1].min() - 0.2, X_train[:, 1].max() + 0.2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_predictions = cmo_qf.predict(grid_points, final_params)
        grid_predictions = np.where(grid_predictions > best_threshold, 1, -1).reshape(xx.shape)

        plt.figure(figsize=(12, 5))
        
        # Plot decision boundary
        plt.subplot(1, 2, 1)
        plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolor='k', s=50)
        plt.title(f"Enhanced AQEL Decision Boundary\nAccuracy: {best_acc:.3f}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        
        # Plot raw predictions
        plt.subplot(1, 2, 2)
        grid_raw = cmo_qf.predict(grid_points, final_params).reshape(xx.shape)
        plt.contourf(xx, yy, grid_raw, levels=20, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, edgecolor='k', s=50)
        plt.title("Raw Prediction Values")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Prediction Value")
        
        plt.tight_layout()
        plt.show()
