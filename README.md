# LSTR-lane-detect-onnxrun-cpp-py
使用ONNXRuntime部署LSTR基于Transformer的端到端实时车道线检测，包含C++和Python两个版本的程序。

onnx文件的大小只有2.93M，可以做到实时性轻量化部署。
起初，我想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错， 无赖只能使用onnxruntime做部署了。
