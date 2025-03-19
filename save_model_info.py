import json
import numpy as np
import os

def save_model_info(model, metrics, epochs, learning_rates, file_prefix="/Users/yavuzalpdemirci/Documents/SYZ Teknofest/project_main_v2/save_metrics/"):
    """Save simplified model information including metrics, learning rate, and epochs."""
    info_file = f"{file_prefix}_model_info.json"
    
    # Convert all NumPy arrays inside metrics to lists and remove None values
    metrics_serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
        if value is not None
    }
    
    # Simplified model summary
    model_summary = {
        "layers": len(model.layers),
        "trainable_params": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
        "non_trainable_params": int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    }
    
    # Collecting all data
    model_info = {
        "epochs": epochs,
        "initial_learning_rate": learning_rates[0],
        "final_learning_rate": learning_rates[1],
        "metrics": metrics_serializable,
        "model_summary": model_summary
    }
    
    # Append or create new file
    if os.path.exists(info_file):
        with open(info_file, "r+") as json_file:
            existing_data = json.load(json_file)
            existing_data.append(model_info)
            json_file.seek(0)
            json.dump(existing_data, json_file, indent=4)
    else:
        with open(info_file, "w") as json_file:
            json.dump([model_info], json_file, indent=4)
    
    print("Model info appended successfully.")