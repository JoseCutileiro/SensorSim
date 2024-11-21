import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def ukf(previous_points, prediction_range):
    dt = 1.0
    Q = np.eye(6) * 0.1 
    R = np.eye(2) * 0.5 
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0]) 
    P = np.eye(6) 


    n = len(x)
    alpha = 1e-3
    beta = 2
    kappa = 0
    lambda_ = alpha ** 2 * (n + kappa) - n

    # Calculate weights for sigma points
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.copy(Wm)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha ** 2 + beta)

    # State transition and observation functions
    def f(x):
        """State transition function."""
        return np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt ** 2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt ** 2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])

    def h(x):
        """Observation function."""
        return np.array([x[0], x[1]])

    # Prediction and update for each measurement in previous_points
    for z in previous_points:
        # Generate sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]

        # Prediction step
        sigma_points_pred = np.array([f(point) for point in sigma_points])
        x_pred = np.dot(Wm, sigma_points_pred)
        P_pred = Q + np.sum([Wc[i] * np.outer(sigma_points_pred[i] - x_pred, sigma_points_pred[i] - x_pred)
                             for i in range(2 * n + 1)], axis=0)

        # Update step
        sigma_points_obs = np.array([h(point) for point in sigma_points_pred])
        z_pred = np.dot(Wm, sigma_points_obs)
        S = R + np.sum([Wc[i] * np.outer(sigma_points_obs[i] - z_pred, sigma_points_obs[i] - z_pred)
                        for i in range(2 * n + 1)], axis=0)

        cross_covariance = np.sum([Wc[i] * np.outer(sigma_points_pred[i] - x_pred, sigma_points_obs[i] - z_pred)
                                   for i in range(2 * n + 1)], axis=0)
        K = cross_covariance @ np.linalg.inv(S)
        x = x_pred + K @ (z - z_pred)
        P = P_pred - K @ S @ K.T

    # Prediction phase to generate future points
    predictions = []
    for _ in range(prediction_range):
        # Predict the next state
        sigma_points = np.zeros((2 * n + 1, n))
        sqrt_P = np.linalg.cholesky((n + lambda_) * P)
        sigma_points[0] = x
        for i in range(n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[n + i + 1] = x - sqrt_P[i]

        sigma_points_pred = np.array([f(point) for point in sigma_points])
        x_pred = np.dot(Wm, sigma_points_pred)

        predictions.append(x_pred[:2])
        x = x_pred  # Update x for the next prediction step

    return predictions

# Função do EKF
def ekf(previous_points, prediction_range):
    dt = 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(2) * 0.5
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0, 0, 0])
    P = np.eye(6)
    F_jacobian = np.array([
        [1, 0, dt, 0, 0.5 * dt**2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    for z in previous_points:
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        P = F_jacobian @ P @ F_jacobian.T + Q
        y = z - H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(len(K)) - K @ H) @ P

    predictions = []
    for _ in range(prediction_range):
        x_pred = np.array([
            x[0] + x[2] * dt + 0.5 * x[4] * dt**2,
            x[1] + x[3] * dt + 0.5 * x[5] * dt**2,
            x[2] + x[4] * dt,
            x[3] + x[5] * dt,
            x[4],
            x[5]
        ])
        predictions.append(x_pred[:2])
        x = x_pred

    return predictions

def linear_regression_predictor(previous_points, prediction_range):
    # Convert points to numpy array for easier manipulation
    previous_points = np.array(previous_points)
    n_points = len(previous_points)

    # Time steps for previous points (assuming uniform time intervals of 1)
    t = np.arange(n_points).reshape(-1, 1)

    # Separate x and y coordinates
    x_coords = previous_points[:, 0]
    y_coords = previous_points[:, 1]

    # Fit linear regression for x and y separately
    x_model = LinearRegression().fit(t, x_coords)
    y_model = LinearRegression().fit(t, y_coords)

    # Generate future time steps
    future_t = np.arange(n_points, n_points + prediction_range).reshape(-1, 1)

    # Predict future x and y values
    x_predictions = x_model.predict(future_t)
    y_predictions = y_model.predict(future_t)

    # Combine x and y predictions into a list of coordinate pairs
    predictions = np.column_stack((x_predictions, y_predictions))

    return predictions.tolist()

def polynomial_regression_predictor(previous_points, prediction_range):
    # Convert points to numpy array for easier manipulation
    previous_points = np.array(previous_points)
    n_points = len(previous_points)

    # Time steps for previous points (assuming uniform time intervals of 1)
    t = np.arange(n_points).reshape(-1, 1)

    # Separate x and y coordinates
    x_coords = previous_points[:, 0]
    y_coords = previous_points[:, 1]

    # Use PolynomialFeatures to add a quadratic term (for acceleration)
    poly = PolynomialFeatures(degree=2)
    t_poly = poly.fit_transform(t)

    # Fit polynomial regression for x and y separately
    x_model = LinearRegression().fit(t_poly, x_coords)
    y_model = LinearRegression().fit(t_poly, y_coords)

    # Generate future time steps and transform them for quadratic prediction
    future_t = np.arange(n_points, n_points + prediction_range).reshape(-1, 1)
    future_t_poly = poly.transform(future_t)

    # Predict future x and y values
    x_predictions = x_model.predict(future_t_poly)
    y_predictions = y_model.predict(future_t_poly)

    # Combine x and y predictions into a list of coordinate pairs
    predictions = np.column_stack((x_predictions, y_predictions))

    return predictions.tolist()

# Função do Kalman Filter
def kalman_filter(previous_points, prediction_range):
    dt = 1.0  # Intervalo de tempo entre as medições
    Q = np.eye(4) * 0.1  # Covariância do ruído do processo
    R = np.eye(2) * 0.5  # Covariância do ruído de medição

    # Inicialização do estado [x, y, vx, vy]
    x = np.array([previous_points[0][0], previous_points[0][1], 0, 0])  
    P = np.eye(4)  # Covariância inicial

    # Matriz de transição de estados (F)
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Matriz de observação (H)
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    predictions = []

    # Passo de atualização com as medições
    for z in previous_points:
        # Predição
        x_pred = F @ x
        P = F @ P @ F.T + Q

        # Medição
        y = np.array(z) - H @ x_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        # Atualização do estado e covariância
        x = x_pred + K @ y
        P = (np.eye(len(K)) - K @ H) @ P

    # Passo de predição futura
    for _ in range(prediction_range):
        x_pred = F @ x
        predictions.append(x_pred[:2])  # Apenas as coordenadas x, y são necessárias
        x = x_pred

    return predictions