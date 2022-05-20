import logging

import pandas as pd
import argparse
import pathlib
import json
import os
import numpy as np
import tarfile
import uuid

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_path = "/opt/ml/processing/model" #"model" # 
output_path = '/opt/ml/processing/output' #"output" # 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    args, _ = parser.parse_known_args()

    
    zero = [0]
    one = [1]
    
    class_name_list = ['Fraud', 'Not Fraud']
    
    acts = 500*zero + 10*one

    preds = 502*zero + 8*one
    
    precision = precision_score(acts, preds, average='micro')
    recall = recall_score(acts, preds, average='micro')
    accuracy = accuracy_score(acts, preds)
    cnf_matrix = confusion_matrix(acts, preds, labels=range(len(class_name_list)))
    f1 = f1_score(acts, preds, average='micro')
    
    print("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(cnf_matrix))
    logger.debug("F1 score: {}".format(f1))
    
    print(cnf_matrix)
    
    matrix_output = dict()
    
    for i in range(len(cnf_matrix)):
        matrix_row = dict()
        for j in range(len(cnf_matrix[0])):
            matrix_row[j] = int(cnf_matrix[i][j])
        matrix_output[i] = matrix_row

    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "confusion_matrix":matrix_output
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))