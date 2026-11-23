FROM public.ecr.aws/lambda/python:3.12-arm64

RUN pip install --no-cache-dir \
    "onnxruntime==1.32.0" \
    "tokenizers==0.21.0" \
    "numpy==2.2.3" \
    "psycopg[binary]==3.2.6" \
    "cryptography==44.0.2" \
    "boto3>=1.35"

COPY clync_cipher.py ${LAMBDA_TASK_ROOT}/
COPY handler.py ${LAMBDA_TASK_ROOT}/
COPY model/ ${LAMBDA_TASK_ROOT}/model/

CMD ["handler.lambda_handler"]
