#!/bin/bash
# get google drive file as per https://stackoverflow.com/a/58914589

# Get preprocessed subset of ShapeNet data (370.96MB) (https://drive.google.com/file/d/14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0/view?usp=sharing)
curl -L "https://drive.usercontent.google.com/download?id=14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0&confirm=xxx" -o NSP_data.tar.gz
tar -xzvf NSP_data.tar.gz

# Get .ply point tree files for ShapeNet subset (412.80MB) (https://drive.google.com/file/d/1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf/view?usp=sharing)
curl -L "https://drive.usercontent.google.com/download?id=1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf&confirm=xxx" -o ShapeNetNSP.tar.gz
tar -xzvf ShapeNetNSP.tar.gz