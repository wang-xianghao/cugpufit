import cunumeric as np

from model import Model
from losses.mse import MSE

def compute_gauss_newton(J, JJ, rhs):
    updates = np.linalg.solve(JJ, rhs) # TODO: more options for solving JJ^Tx=rhs
    updates = J.T @ updates

def _fit_batch(model: Model, inputs, targets):
    N, D = inputs.shape
    normalization_factor = np.divide(1.0, D, dtype=model.dtype)
    
    outputs = model.predict(inputs, train=True)
    J = model.jacobian
    loss, residuals = MSE().loss_with_residuals(inputs, targets)
    
    # Gauss-Newton underdetermined init
    # TODO: Gauss-Newton overdetermined init
    JJ = J.T @ J
    rhs = residuals
    
    JJ *= normalization_factor
    rhs *= normalization_factor
    
    # TODO: implement damping algorithm   
    damping = 0.0
    JJ_damped = JJ
    
    updates = compute_gauss_newton(J, JJ_damped, rhs)
    model.update(updates)

def fit(model, inputs, targets, max_n_epochs, batch_size):
    assert inputs.shape[0] % batch_size == 0
    
    all_batch_inputs = np.split(inputs, batch_size)
    all_batch_targets = np.split(targets, batch_size)
    
    for i in range(max_n_epochs):
        print(f'Training epoch {i + 1}')
        for batch_inputs, batch_targets in zip(all_batch_inputs, all_batch_targets):
            _fit_batch(model, batch_inputs, batch_targets)