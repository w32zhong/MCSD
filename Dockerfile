FROM nvcr.io/nvidia/pytorch:23.11-py3
WORKDIR /workspace
ADD ./LLM-common-eval/requirements.txt r1.txt
RUN pip install -r r1.txt
ADD ./mcsd/MCSD/requirements.txt r2.txt
RUN pip install -r r2.txt
RUN pip install transformers==4.34.1
RUN pip install huggingface-hub==0.19.4
ADD . s3d
WORKDIR /workspace/s3d
CMD /bin/bash
