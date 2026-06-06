import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import dagshub

def parse_args():
    parser = argparse.ArgumentParser(description="Train German Credit RandomForest Model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Max depth of tree")
    return parser.parse_args()

def setup_mlflow():
    dagshub_username = os.environ.get("DAGSHUB_USERNAME")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    
    if dagshub_username and dagshub_token:
        print("CI Mode: Setting remote tracking URI for DagsHub.")
        mlflow.set_tracking_uri("https://dagshub.com/charnom/MSML-Submission.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    else:
        print("Local Mode: Setting local tracking URI.")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
    mlflow.set_experiment("Latihan CI MLProject")

def train():
    args = parse_args()
    setup_mlflow()
    
    # Load preprocessed datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'namadataset_preprocessing', 'train_preprocessed.csv')
    test_path = os.path.join(base_dir, 'namadataset_preprocessing', 'test_preprocessed.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Preprocessed train/test data not found. Ensure namadataset_preprocessing contains correct files.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['Risk'])
    y_train = train_df['Risk']
    X_test = test_df.drop(columns=['Risk'])
    y_test = test_df['Risk']
    
    print(f"Training RandomForest(n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
    with mlflow.start_run(run_name="CI_MLProject_Run"):
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        
        # Infer model signature (Inputs and Outputs schema)
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log model with signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="GermanCreditCIModel"
        )
        
        print("Model training and logging completed successfully.")

if __name__ == '__main__':
    train()
