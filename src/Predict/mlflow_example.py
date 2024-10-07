import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
mlflow.set_tracking_uri("http://127.0.0.1:8765")
# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow tracking
mlflow.start_run()

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Log model, parameters, and metrics
mlflow.log_param("model_type", "LinearRegression")
mlflow.log_metric("mse", mse)
mlflow.log_metric("r2", r2)
mlflow.sklearn.log_model(model, "model")

# End MLflow run
mlflow.end_run()

print(f"Logged model with R2: {r2} and MSE: {mse}")
