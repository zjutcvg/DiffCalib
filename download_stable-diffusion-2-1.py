#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/stable-diffusion-2-1',cache_dir='./checkpoint')