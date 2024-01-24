from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,StableDiffusionPipeline
import torch
from torchinfo import  summary
from PIL import Image
from p2p_attention_controller import AttentionStore
from p2p_attention_controller import show_cross_attention
from p2p_attention_controller import AttentionReplace
from p2p_attention_controller import AttentionRefine
import p2p_attention_controller
import ptp_utils
import time
import pickle
from multiprocessing import shared_memory
from pympler import asizeof
from p2p_attention_controller import get_equalizer,AttentionReweight

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


start_time=time.time()
pipe = StableDiffusionXLPipeline.from_single_file(
    pretrained_model_link_or_path=r"C:\App\comfyui\HoloAnime-XLv27.safetensors",original_config_file=r"C:\code\stable-diffusion-webui\configs\sd_xl_base.yaml",local_files_only=True, torch_dtype=torch.float16).to("cuda")
end_time=time.time()
print("read file time:",end_time-start_time)
tokenizer=pipe.tokenizer
Width=1024
Height=1024

### 无控制双图
# prompt=["a furry bear riding a bicycle"]
# negative_prompt=["blurred,flaw,defect,blemish,fault,disproportionate,distorted,uncoordinated,ugly,bad,blur,dumb,stubborn,noise dots,cheap,inferior,prototype,unfinished,rough draft,preliminary,sketch,lack,ordinary,abrupt,photo,realistic,3d model,bad,worse,worst,ugly,bad anatomy, blurry, close-up,disembodied limb,cropped,deformed,text,icon,artist name,signature,twitter username,monochrome,disfigured,glitch"]
# start_time=time.time()
# image,latent = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     width=Width,
#     height=Height,
#     guidance_scale=7,
#     target_size=(1024,1024),
#     # original_size=(4096,4096),
#     num_inference_steps=25,
#     controller=p2p_attention_controller.EmptyControl(),
#     # controller=controller,
#     )
# image[0][0].save('normal_output/output_image1.png')
# prompt=["a furry dog riding a bicycle"]
# image,latent2 = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     width=Width,
#     height=Height,
#     guidance_scale=7,
#     target_size=(1024,1024),
#     # original_size=(4096,4096),
#     num_inference_steps=25,
#     controller=p2p_attention_controller.EmptyControl(),
#     latents=latent,
#     # controller=controller,
#     special_latent=latent
#     )
# print(latent.shape)
# image[0][0].save('normal_output/output_image2.png')
# breakpoint()

##### 注意力图可视化部分
controller = AttentionStore()
prompt=["a bear riding bicycle in forest"]
negative_prompt=["blurred,flaw,defect,blemish,fault,disproportionate,distorted,uncoordinated,ugly,bad,blur,dumb,stubborn,noise dots,cheap,inferior,prototype,unfinished,rough draft,preliminary,sketch,lack,ordinary,abrupt,photo,realistic,3d model,bad,worse,worst,ugly,bad anatomy, blurry, close-up,disembodied limb,cropped,deformed,text,icon,artist name,signature,twitter username,monochrome,disfigured,glitch"]
start_time=time.time()
image,latent = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=Width,
    height=Height,
    guidance_scale=7,
    target_size=(1024,1024),
    # original_size=(4096,4096),
    num_inference_steps=25,
    # controller=p2p_attention_controller.EmptyControl(),
    controller=controller,
    )
image[0][0].save('output_image.png')

pipe=pipe.to('cpu')
del pipe
torch.cuda.empty_cache()
show_cross_attention(controller, res=32, from_where=("up", "down"),tokenizer=tokenizer,prompts=prompt)
end_time=time.time()
print("生图用时:",end_time-start_time)

stats = torch.cuda.memory_stats(device=torch.cuda.current_device())
sizes=0
for key in controller.attention_store.keys():
    for map in controller.attention_store[key]:
        size=ptp_utils.cal_memory(map)
        sizes+=size
print("controller.attention_store:",sizes)
for key in controller.attention_store.keys():
    for i,map in enumerate(controller.attention_store[key]):
        controller.attention_store[key][i]=controller.attention_store[key][i].to('cpu')
del controller
torch.cuda.empty_cache()
breakpoint()
### prompt替换
# prompts = ["a beautiful girl with light smile,manga style,simple background",
#            "a beautiful girl with angry face,manga style,simple background"]
# negative_prompt=2*["blurred,flaw,defect,blemish,fault,disproportionate,distorted,uncoordinated,ugly,bad,blur,dumb,stubborn,noise dots,cheap,inferior,prototype,unfinished,rough draft,preliminary,sketch,lack,ordinary,abrupt,photo,realistic,3d model,bad,worse,worst,ugly,bad anatomy, blurry, close-up,disembodied limb,cropped,deformed,text,icon,artist name,signature,twitter username,monochrome,disfigured,glitch"]
# if prompts is not None and isinstance(prompts, str):
#     batch_size = 1
# elif prompts is not None and isinstance(prompts, list):
#     batch_size = len(prompts)
# # latents = pipe.prepare_latents(
# #     batch_size * 1,
# #     num_channels_latents = pipe.unet.config.in_channels,
# #     height=Height,
# #     width=Width,
# #     dtype=torch.float16,
# #     device='cuda:0',
# #     generator=None,
# #     latents=None,
# # )
#
# controller = AttentionReplace(prompts, num_steps=25, cross_replace_steps=0.4, self_replace_steps=0.5,tokenizer=tokenizer,device="cuda:0")#cross_replace_steps=0.8，代表在扩散了80%再将注意力权重替换为新的
# images,latent = pipe(
#     prompts,
#     negative_prompt=negative_prompt,
#     width=Width,
#     height=Height,
#     guidance_scale=7,
#     target_size=(1024,1024),
#     # original_size=(2048,2048),
#     num_inference_steps=25,
#     controller=controller,
#     # controller=p2p_attention_controller.EmptyControl(),
#     #latents=latents
#     )
#
# path='replacing/'
# images[0][0].save(f'{path}{prompts[0]}1.png')
# images[0][1].save(f'{path}{prompts[0]}2_control.png')
#
# breakpoint()
### 权重更改

# prompts = ["a furry bear","a furry bear"]
# negative_prompt=["",""]
# ### pay 3 times more attention to the word "smiling"
# equalizer = get_equalizer(prompts[1], ("furry",), (10,),tokenizer=tokenizer)
# # equalizer:tensor([[1., 1., 5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
# #          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
# #          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
# #          1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
# #          1., 1., 1., 1., 1.]])#[1,77]
#
# controller = AttentionReweight(prompts, 25, cross_replace_steps=0.9,
#                                self_replace_steps=0.2,
#                                equalizer=equalizer)
#
# images,latent = pipe(
#     prompts,
#     negative_prompt=negative_prompt,
#     width=Width,
#     height=Height,
#     guidance_scale=7,
#     target_size=(1024,1024),
#     # original_size=(2048,2048),
#     num_inference_steps=25,
#     controller=controller,
#     #controller=p2p_attention_controller.EmptyControl(),
#     #latents=latents
#     )
# path='reweighting/'
# images[0][0].save(f'{path}{prompts[0]}1.png')
# images[0][1].save(f'{path}{prompts[0]}2.png')

# ### 添加新元素
#
prompts = ["a cute cat",
           "a cute cat,on the street"]
controller = AttentionRefine(prompts, 25,
                             cross_replace_steps=0.2,
                             self_replace_steps=0.3,tokenizer=tokenizer)
images,latent0 = pipe(
    prompts,
    # negative_prompt=negative_prompt,
    width=Width,
    height=Height,
    guidance_scale=7,
    target_size=(1024,1024),
    # original_size=(2048,2048),
    num_inference_steps=25,
    controller=controller,
    # controller=p2p_attention_controller.EmptyControl(),
    #latents=latents
    )

path='refining/'
images[0][0].save(f'{path}{prompts[0]}1.png')
images[0][1].save(f'{path}{prompts[1]}2.png')

prompts = ["a cute cat",
           "a cute cat,with clothes"]
controller = AttentionRefine(prompts, 25,
                             cross_replace_steps=0.2,
                             self_replace_steps=0.3,tokenizer=tokenizer)
images,latent1 = pipe(
    prompts,
    # negative_prompt=negative_prompt,
    width=Width,
    height=Height,
    guidance_scale=7,
    target_size=(1024,1024),
    # original_size=(2048,2048),
    num_inference_steps=25,
    controller=controller,
    # controller=p2p_attention_controller.EmptyControl(),
    special_latent=latent0
    )

path='refining/'
images[0][0].save(f'{path}{prompts[0]}1.png')
images[0][1].save(f'{path}{prompts[1]}2.png')

prompts = ["a cute cat",
           "a cute cat,style of impressionism"]
controller = AttentionRefine(prompts, 25,
                             cross_replace_steps=0.2,
                             self_replace_steps=0.2,tokenizer=tokenizer)
images,latent1 = pipe(
    prompts,
    # negative_prompt=negative_prompt,
    width=Width,
    height=Height,
    guidance_scale=7,
    target_size=(1024,1024),
    # original_size=(2048,2048),
    num_inference_steps=25,
    controller=controller,
    # controller=p2p_attention_controller.EmptyControl(),
    special_latent=latent0
    )

path='refining/'
images[0][0].save(f'{path}{prompts[0]}1.png')
images[0][1].save(f'{path}{prompts[1]}2.png')




