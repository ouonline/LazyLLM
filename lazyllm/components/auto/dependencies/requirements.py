requirements = dict(alpacalora='''
accelerate>=0.30.1
appdirs
loralib>=0.1.1
bitsandbytes
datasets
fire
peft>=0.3.0
transformers>=4.41.1
sentencepiece>=0.1.99
gradio
tokenizers>=0.19.1
scipy
deepspeed>=0.12.3
sentence_transformers
faiss-cpu
''', collie='''
collie-lm>=1.0.7
accelerate>=0.30.1
appdirs
loralib>=0.1.1
bitsandbytes
datasets
fire
peft>=0.3.0
transformers>=4.41.1
sentencepiece>=0.1.99
gradio
tokenizers>=0.19.1
scipy
deepspeed>=0.12.3
sentence_transformers
faiss-cpu
''', llamafactory='''
llamafactory
accelerate
datasets
einops
fastapi
fire
gradio
matplotlib
numpy
packaging
pandas
peft
protobuf
pydantic
pyyaml
scipy
sentencepiece
sse-starlette
tiktoken
transformers
trl
uvicorn
''', lightllm='''
lightllm>=0.0.1
einops
ninja
packaging
pyzmq
rpyc
safetensors
torch>=2.1.2
transformers>=4.41.1
triton>=2.1.0
uvloop
''', vllm='''
vllm>=0.4.0
cmake
fastapi
ninja
numpy
outlines
prometheus-client
psutil
py-cpuinfo
pydantic
pynvml
ray
requests
sentencepiece
tiktoken
torch>=2.1.2
transformers>=4.41.1
triton>=2.1.0
uvicorn
xformers
''')
