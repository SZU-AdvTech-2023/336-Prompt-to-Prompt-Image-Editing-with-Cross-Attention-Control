from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from diffusers import StableDiffusionXLPipeline
import matplotlib.pyplot as plt

LOW_RESOURCE=0

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 32, 32, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                self.forward(attn, is_cross, place_in_unet)
            else:
                print("call attn shape:",attn.shape)
                h = attn.shape[0]
                # attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

        #计算 self.attention_store 的存储空间
        # sizes=0
        # for key in self.attention_store.keys():
        #     for map in self.attention_store[key]:
        #         size=ptp_utils.cal_memory(map)
        #         sizes+=size
        # print("size of attention_store:",sizes)
        #type of attention_store[key] is list
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            # size=ptp_utils.cal_memory(attn)
            print("forward attn shape:", attn.shape)
            # print("attn size:",size)
            attn_copy=attn.clone().detach()
            self.step_store[key].append(attn_copy)#将每一步得到的注意力图存入到对应位置的列表中. 似乎就是这句代码导致显存大幅占用
            # print("self.step_store[key] attn counts:",len(self.step_store[key]))
        sizes = 0
        for key in self.step_store.keys():
            for map in self.step_store[key]:
                size = ptp_utils.cal_memory(map)
                sizes += size
        print("self.step_store:", sizes)
        return 0

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]#每一步逐渐累加

        #尝试减少显存使用,并没什么用
        # for key in self.step_store.keys():
        #     for i in range(len(self.step_store[key])):
        #         self.step_store[key][i].to('cpu')
        # del self.step_store
        # torch.cuda.empty_cache()

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}#对注意力图的总和除以其步数
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):

        if att_replace.shape[2] <= 32 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):#attn来自self.get_attention_scores(query, key, attention_mask)
        # super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet) #意义不明的调用
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):

            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]#多张图同时进行运算，
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]#控制替换与否,self.cross_replace_alpha:torch.Size([21, 1, 1, 1, 77])
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)#将selfattention的注意力图进行替换
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer=None,
                 device="cuda:0"):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps#经过这行代码后，如果 self_replace_steps 原本是一个浮点数（例如 0.5），它将被转换为一个元组，其内容为 (0, 0.5)
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend



class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        #[20, 1024, 77],[1, 77, 77]->[1,20,1024,77]
        #attn_base和torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)数值相同
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: torch.float16, self_replace_steps: torch.float16,
                 local_blend: Optional[LocalBlend] = None,
                 tokenizer=None,
                 device="cuda:0"):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)#
        #调用父类构造函数：super(AttentionReplace, self).__init__(...) 调用了 AttentionControlEdit 类的构造函数。这意味着，当创建 AttentionReplace 类的实例时，首先会按照 AttentionControlEdit 类的构造函数初始化一些基本属性或执行一些基本操作。
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)#torch.Size([1, 77, 77])


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,tokenizer=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        print(self.mapper)
        self.mapper, alphas = self.mapper.to('cuda:0'), alphas.to('cuda:0')
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        #attn_base.shape torch.Size([10, 4096, 77])
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        print(self.equalizer)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to('cuda:0')
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                                                                                     Tuple[float, ...]],tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    values = torch.tensor(values, dtype=torch.float32)
    for i,word in enumerate(word_select):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values[i]
    return equalizer


from PIL import Image


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int,prompts=[""]):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:#读出各个位置的注意力图,torch.Size([20, 1024, 77])
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)#torch.Size([1000, 32, 32, 77])
    out = out.sum(0) / out.shape[0]#将各个位置的注意力图归一化
    return out.cpu()



def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,tokenizer=None,prompts=[""]):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select,prompts=prompts)#关键函数-获取注意力权重图
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = image.numpy().astype(np.float)
        # image = image/image.max()
        image = (image - image.min()) / (image.max() - image.min())
        cmap = plt.get_cmap('coolwarm')  # 选择颜色映射
        colored_image = np.zeros((32, 32, 4), dtype=np.float)
        for j in range(32):
            for k in range(32):
                colored_image[j, k] = cmap(image[j, k])
        mapped_data_rgb = (colored_image[:, :, :3] * 255).astype(np.uint8)
        image = np.array(Image.fromarray(mapped_data_rgb).resize((256, 256)))#(256, 256, 3)
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))
#
#
def run_and_display(prompts,negative_prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts,negative_prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    ptp_utils.view_images(images)
    return images, x_t

if __name__=="__main__":
    print(1)
    breakpoint()
    LOW_RESOURCE = False
    NUM_DIFFUSION_STEPS = 20
    GUIDANCE_SCALE = 7
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = StableDiffusionXLPipeline.from_pretrained("HoloAnime",torch_dtype=torch.float16).to(device)
    tokenizer = ldm_stable.tokenizer

    g_cpu = torch.Generator().manual_seed(8888)
    prompts = ["a beautiful girl with blue hair wearing school uniform,full body,manga style,hand up"]
    negative_prompts=[""]
    controller = AttentionStore()
    image, x_t = run_and_display(prompts,negative_prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
    show_cross_attention(controller, res=16, from_where=("up", "down"))