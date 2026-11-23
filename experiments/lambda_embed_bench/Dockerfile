FROM public.ecr.aws/lambda/python:3.12-arm64

RUN pip install --no-cache-dir \
    onnxruntime==1.32.0 \
    tokenizers==0.21.0 \
    numpy==2.2.3

COPY model/bge_m3_int8_dense_sparse.onnx /var/task/model/bge_m3_int8_dense_sparse.onnx
COPY model/tokenizer.json /var/task/model/tokenizer.json
COPY lambda_handler.py /var/task/lambda_handler.py

CMD ["lambda_handler.handler"]
