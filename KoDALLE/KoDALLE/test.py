import torch
import torchvision.transforms as T
from dalle_pytorch import VQGanVAE
from dalle.models import DALLE_Klue_Roberta
from transformers import AutoTokenizer

import yaml
from easydict import EasyDict


dalle_config_path = 'configs/dalle_config.yaml'
dalle_path = '/mnt/hdd1/wkim/DS503/KoDALLE/results/dalle_uk.pt'

vqgan_config_path = '/mnt/hdd1/wkim/DS503/taming-transformers/configs/VQGAN_blue.yaml'
vqgan_path = '/mnt/hdd1/wkim/DS503/taming-transformers/logs/2023-05-24T11-25-54_VQGAN_blue/checkpoints/epoch=000020.ckpt'
# vqgan_path = 'vqgan/vae-final-v2.pt'

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)


tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

with open(dalle_config_path, "r") as f:
    dalle_config = yaml.load(f, Loader=yaml.Loader)
    DALLE_CFG = EasyDict(dalle_config["DALLE_CFG"])

DALLE_CFG.VOCAB_SIZE = tokenizer.vocab_size

vae = VQGanVAE(
    vqgan_model_path=vqgan_path, 
    vqgan_config_path=vqgan_config_path
)

DALLE_CFG.IMAGE_SIZE = vae.image_size

dalle_params = dict(
    num_text_tokens=tokenizer.vocab_size,
    text_seq_len=DALLE_CFG.TEXT_SEQ_LEN,
    depth=DALLE_CFG.DEPTH,
    heads=DALLE_CFG.HEADS,
    dim_head=DALLE_CFG.DIM_HEAD,
    reversible=DALLE_CFG.REVERSIBLE,
    loss_img_weight=DALLE_CFG.LOSS_IMG_WEIGHT,
    attn_types=DALLE_CFG.ATTN_TYPES,
    ff_dropout=DALLE_CFG.FF_DROPOUT,
    attn_dropout=DALLE_CFG.ATTN_DROPOUT,
    stable=DALLE_CFG.STABLE,
    shift_tokens=DALLE_CFG.SHIFT_TOKENS,
    rotary_emb=DALLE_CFG.ROTARY_EMB,
)

dalle = DALLE_Klue_Roberta(
    vae=vae, 
    wte_dir="models/roberta_large_wte.pt",
    wpe_dir="models/roberta_large_wpe.pt",
    **dalle_params
    ).to(device)


loaded_obj = torch.load(dalle_path, map_location=torch.device('cuda:0'))
dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
dalle.load_state_dict(weights)


def text_to_montage(text):
    encoded_dict = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=DALLE_CFG.TEXT_SEQ_LEN,
        add_special_tokens=True,
        return_token_type_ids=True,  # for RoBERTa
    ).to(device)

    encoded_text = encoded_dict['input_ids']
    mask = encoded_dict['attention_mask']

    image = dalle.generate_images(
        encoded_text,
        mask=mask,
        filter_thres=0.9  # topk sampling at 0.9
    )

    return T.ToPILImage()(image.squeeze())


# text = '볼이 넓은 계란형 얼굴이며 앞머리가 이마의 양쪽 끝을 가리고 있어 모양은 보이지 않는다.   오른쪽 턱의 각진 부분이 왼쪽에 비해 아래로 내려와 있고 왼쪽은 약간 완만한 형태이다.  턱끝으로 내려오는 턱모양은 약간 둥근형으로 보인다. 왼쪽의 볼이 더 평평하고 넓은 편이다.'
# text = '20대 후반. 날카로운 눈매. 날카로운 턱'

# text = '40대 후반 남성, 얼굴이 길고 눈썹이 짙다. 넓은 이마, 단정한 머리. 무표정.'
# text = '날카로운 눈매에 정갈한 머리 스타일, 긴 두상, 오똑한 코에 큰 귀'
# text = '중년남성, 눈썹 두꺼움, 쌍커풀 없음, 무섭게 생김.'
text = '50대 남성, 두꺼운 눈썹, 숱이 많고, 이마가 좁고, 코가 둥글고, 인상이 무섭고, 입이 작다.'
image = text_to_montage(text)

image.save('./images/image_yj.png')