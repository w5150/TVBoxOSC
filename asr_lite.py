import sys, json, re, os, torch, torch.nn as nn, urllib.request, hashlib
from funasr import AutoModel
from funasr.models.lora.layers import Linear as LoRALinear

# ===== 配置 =====
MODEL_NAME = 'iic/SenseVoiceSmall'
ADAPTER_FILE = os.path.join(os.path.dirname(__file__), 'sensevoice_lora_adapter.pt')
ADAPTER_URL = 'https://github.com/w5150/TVBoxOSC/releases/download/1.1/sensevoice_lora_adapter.pt'
EXPECTED_SHA256 = '36f39789c9b2a1bfa357fc46970bfb7f0f032d7e853126eb7abad029460ab143'
# ================

if not os.path.exists(ADAPTER_FILE):
    if '你的用户名' in ADAPTER_URL:
        raise RuntimeError('请先修改 ADAPTER_URL 为你的 GitHub Release 地址')
    print(f'Downloading LoRA adapter from GitHub...', file=sys.stderr, flush=True)
    urllib.request.urlretrieve(ADAPTER_URL, ADAPTER_FILE + '.tmp')
    if EXPECTED_SHA256:
        actual = hashlib.sha256(open(ADAPTER_FILE + '.tmp', 'rb').read()).hexdigest()
        if actual != EXPECTED_SHA256:
            os.remove(ADAPTER_FILE + '.tmp')
            raise RuntimeError(f'SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}')
    os.rename(ADAPTER_FILE + '.tmp', ADAPTER_FILE)
    print(f'Downloaded {ADAPTER_FILE}', file=sys.stderr, flush=True)

print('Loading model...', file=sys.stderr, flush=True)
model_obj = AutoModel(model=MODEL_NAME, device='cpu', disable_update=False)
model = model_obj.model

def _apply_lora(module, r=8, lora_alpha=16, depth=0):
    if depth > 100: return
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
            new = LoRALinear(child.in_features, child.out_features, r=r, lora_alpha=lora_alpha, bias=child.bias is not None)
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None: new.bias.data.copy_(child.bias.data)
            setattr(module, child_name, new)
        else:
            _apply_lora(child, r, lora_alpha, depth+1)

_apply_lora(model.encoder)
ckpt = torch.load(ADAPTER_FILE, map_location='cpu')
model.load_state_dict(ckpt['lora_state_dict'], strict=False)
model.load_state_dict(ckpt['ctc_state_dict'], strict=False)
model.eval()
print('Ready', file=sys.stderr, flush=True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: python asr_lite.py <wav_path>'}, ensure_ascii=False))
        sys.exit(1)
    wav = sys.argv[1]
    if not os.path.exists(wav):
        print(json.dumps({'error': f'File not found: {wav}'}, ensure_ascii=False))
        sys.exit(1)
    res = model_obj.generate(input=wav, language='zh', use_itn=True)
    text = re.sub(r'<\|[^>]*\|>', '', res[0]['text']).strip()
    print(json.dumps({'text': text}, ensure_ascii=False))
