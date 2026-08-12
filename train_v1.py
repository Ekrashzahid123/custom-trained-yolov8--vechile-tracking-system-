import os
import json
from ultralytics import YOLO

def train_and_eval_v1():
    dataset_yaml = r'd:\Model Imrovement\Vehicle-counting.v1i.yolov8\data.yaml'
    project_dir = r'd:\Model Imrovement\runs'
    name_v1 = 'v1_baseline'
    
    print("=== Training Model V1 (Baseline) ===")
    model_v1 = YOLO('yolov8n.pt')
    
    # Train V1 for 20 epochs
    train_results = model_v1.train(
        data=dataset_yaml,
        epochs=20,
        batch=16,
        imgsz=416,
        workers=2,
        project=project_dir,
        name=name_v1,
        exist_ok=True,
        seed=42,
        verbose=True
    )
    
    # Best model weights path
    best_weights = os.path.join(project_dir, name_v1, 'weights', 'best.pt')
    print(f"Loading best V1 weights from {best_weights} for evaluation...")
    model_v1_best = YOLO(best_weights)
    
    # Evaluate on validation set
    val_results = model_v1_best.val(
        data=dataset_yaml,
        split='val',
        batch=16,
        imgsz=416,
        project=project_dir,
        name='v1_val_eval',
        exist_ok=True,
        save_json=True
    )
    
    # Evaluate on test set
    test_results = model_v1_best.val(
        data=dataset_yaml,
        split='test',
        batch=16,
        imgsz=416,
        project=project_dir,
        name='v1_test_eval',
        exist_ok=True,
        save_json=True
    )
    
    metrics_summary = {
        'val': {
            'precision': float(val_results.results_dict['metrics/precision(B)']),
            'recall': float(val_results.results_dict['metrics/recall(B)']),
            'map50': float(val_results.results_dict['metrics/mAP50(B)']),
            'map50_95': float(val_results.results_dict['metrics/mAP50-95(B)'])
        },
        'test': {
            'precision': float(test_results.results_dict['metrics/precision(B)']),
            'recall': float(test_results.results_dict['metrics/recall(B)']),
            'map50': float(test_results.results_dict['metrics/mAP50(B)']),
            'map50_95': float(test_results.results_dict['metrics/mAP50-95(B)'])
        }
    }
    
    metrics_path = os.path.join(project_dir, 'v1_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("=== Model V1 Evaluation Complete ===")
    print("V1 Test Metrics:")
    print(json.dumps(metrics_summary['test'], indent=4))

if __name__ == '__main__':
    train_and_eval_v1()
