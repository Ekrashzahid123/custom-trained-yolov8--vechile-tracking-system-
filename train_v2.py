import os
import json
from ultralytics import YOLO

def train_and_eval_v2():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.join(base_dir, 'Vehicle-counting-augmented', 'data.yaml')
    project_dir = os.path.join(base_dir, 'runs')
    name_v2 = 'v2_augmented'
    
    print("=== Training Model V2 (Augmented & Improved) ===")
    model_v2 = YOLO('yolov8n.pt')
    
    # Train V2 for 15 epochs on augmented dataset with batch=32
    train_results = model_v2.train(
        data=dataset_yaml,
        epochs=15,
        batch=32,
        imgsz=416,
        workers=2,
        project=project_dir,
        name=name_v2,
        exist_ok=True,
        seed=42,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True
    )
    
    # Best model weights path
    best_weights = os.path.join(project_dir, name_v2, 'weights', 'best.pt')
    print(f"Loading best V2 weights from {best_weights} for evaluation...")
    model_v2_best = YOLO(best_weights)
    
    # Benchmark evaluation on the test set
    test_results = model_v2_best.val(
        data=dataset_yaml,
        split='test',
        batch=16,
        imgsz=416,
        project=project_dir,
        name='v2_test_eval',
        exist_ok=True,
        save_json=True
    )
    
    metrics_summary = {
        'test': {
            'precision': float(test_results.results_dict['metrics/precision(B)']),
            'recall': float(test_results.results_dict['metrics/recall(B)']),
            'map50': float(test_results.results_dict['metrics/mAP50(B)']),
            'map50_95': float(test_results.results_dict['metrics/mAP50-95(B)'])
        }
    }
    
    metrics_path = os.path.join(project_dir, 'v2_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    print("=== Model V2 Evaluation Complete ===")
    print("V2 Test Metrics:")
    print(json.dumps(metrics_summary['test'], indent=4))

if __name__ == '__main__':
    train_and_eval_v2()
