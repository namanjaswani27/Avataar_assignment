'''Importing Libraries'''
import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoPipelineForInpainting
from scipy.ndimage import binary_dilation

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    '''
        Loads model from hugging face, we use it to get grounding dino model checkpoints
    '''
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model  


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class CFG:
    '''
        Defines variables used in our code
    '''
    sam_type = "vit_h"
    SAM_MODELS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    device = 'cuda'
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
#     image_path = os.path.join(os.getcwd(), 'fruits.jpg')
#     image_path = '/kaggle/input/avataar/wall hanging.jpg'
#     text_prompt = 'chair'


'''Build models'''
def build_sam():
    checkpoint_url = CFG.SAM_MODELS[CFG.sam_type]
    sam = sam_model_registry[CFG.sam_type]()
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    sam.load_state_dict(state_dict, strict=True)
    sam.to(device = CFG.device)
    sam = SamPredictor(sam)
    return sam


def build_groundingdino():
    ckpt_repo_id = CFG.ckpt_repo_id
    ckpt_filename = CFG.ckpt_filename
    ckpt_config_filename = CFG.ckpt_config_filename
    groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
    return groundingdino



'''Predictions'''
def predict_dino(image_pil, text_prompt, box_threshold, text_threshold, model_groundingdino):
    image_trans = transform_image(image_pil)
    boxes, logits, phrases = predict(model = model_groundingdino,
                                     image = image_trans,
                                     caption = text_prompt,
                                     box_threshold = box_threshold,
                                     text_threshold = text_threshold,
                                     device = CFG.device)
    W, H = image_pil.size
    boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H]) # center cood to corner cood
    return boxes, logits, phrases


def predict_sam(image_pil, boxes, model_sam):
    image_array = np.asarray(image_pil)
    model_sam.set_image(image_array)
    transformed_boxes = model_sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    masks, _, _ = model_sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(model_sam.device),
        multimask_output=False,
    )
    return masks.cpu()


def mask_predict(image_pil, text_prompt, models, box_threshold=0.23, text_threshold=0.25):
    boxes, logits, phrases = predict_dino(image_pil, text_prompt, box_threshold, text_threshold, models[0])
    masks = torch.tensor([])
    if len(boxes) > 0:
        masks = predict_sam(image_pil, boxes, models[1])
        masks = masks.squeeze(1)
    return masks, boxes, phrases, logits

'''Utils'''
def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def draw_image(image_pil, masks, alpha=0.4):
    image = np.asarray(image_pil)
    image = torch.from_numpy(image).permute(2, 0, 1)
    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)


def get_segmented(imape_path, text_prompt):
    
    model_sam = build_sam()
    model_groundingdino = build_groundingdino()
    
    models = [model_groundingdino, model_sam]
    
    image_pil = load_image(imape_path)

    masks, boxes, phrases, logits = mask_predict(image_pil, text_prompt, models, box_threshold=0.3, text_threshold=0.25)
    # output = draw_image(image_pil, masks, alpha=0.4)
    return image_pil, masks

    

def process_image(image_pil, masks, output_path, x_units, y_units, text_prompt):
    
    '''Get masked object and background as two separate images'''
    combined_mask = torch.sum(masks, axis=0)
    combined_mask = np.where(combined_mask[:, :] != 0, True, False)
    mask = np.expand_dims(combined_mask, axis=-1)

    masked_object = image_pil * mask
    background = image_pil * ~mask


    '''Shifts image by x_units and y_units'''
    M = np.float32([[1, 0, x_units], [0, 1, -y_units]])
    shifted_image = cv2.warpAffine(masked_object, M, (masked_object.shape[1] , masked_object.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    masked_shifted_image = np.where(shifted_image[:, :, 0] != 0, True, False)

    '''Load stable diffuser model at checkpoint finetuned for inpainting task'''

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        # "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
        "stabilityai/stable-diffusion-2-inpainting",torch_dtype=torch.float16)
    pipe.to(CFG.device)
    # pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

    # With Dilation
    structuring_element = np.ones((20, 20, 1), dtype=bool)
    extrapolated_mask = binary_dilation(mask, structure=structuring_element)
    mask_as_uint8 = extrapolated_mask.astype(np.uint8) * 255
    pil_mask = Image.fromarray(mask_as_uint8.squeeze(), mode='L').resize((1024, 1024))
    # pil_mask

    # # Without Dilation
    # pil_background = Image.fromarray(background)
    # mask_as_uint8 = mask.astype(np.uint8) * 255
    # pil_mask = Image.fromarray(mask_as_uint8.squeeze(), mode='L')
    # # pil_mask

    '''Do inpainting on masked locations of original image'''
    prompt = 'a photo of background'
    inpainted_image = pipe(prompt=prompt,  image=image_pil, mask_image=pil_mask, num_inference_steps=100).images[0]
    # inpainted_image

    '''Get composite of shifted object and background inpainted image'''
    pil_shifted_image = Image.fromarray(shifted_image).resize(inpainted_image.size)
    np_shifted_image = np.array(pil_shifted_image)
    masked_shifted_image = np.where(np_shifted_image[:, :, 0] != 0, True, False)
    masked_shifted_image = np.expand_dims(masked_shifted_image, axis=-1)
    inpainted_shifted = np.array(inpainted_image) * ~masked_shifted_image

    shifted_image = cv2.resize(shifted_image, inpainted_image.size)
    output = inpainted_shifted + shifted_image
    output = Image.fromarray(output)
    output = output.resize(image_pil.size)
    # visualize_results(image_pil, output, 'shifted')
    
    segmented_output = draw_image(image_pil, masks, alpha=0.4)
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(image_pil)
    axes[0].set_title('Original Image')

    axes[1].imshow(segmented_output)
    axes[1].set_title(f'{text_prompt} segmented')
    
    axes[2].imshow(output)
    axes[2].set_title(f'{text_prompt} : {x_units} (+x) {y_units} (+y) Shifted')

    for ax in axes:
        ax.axis('off')

    plt.savefig(output_path, bbox_inches='tight')
    plt.show()



def main():
    import os
    import argparse
    home = os.getcwd()
    
    parser = argparse.ArgumentParser(description='Image segmentation script')
    parser.add_argument('--image', type=str, help='Path to the input image', required=True)
    parser.add_argument('--class_name', type=str, help='Class name for segmentation', required=True)
    parser.add_argument('--output', type=str, help='Path to the output segmented image', required=True)
    parser.add_argument('--x', type=int, help='shift object by +x', required=True)
    parser.add_argument('--y', type=int, help='shift object by +y', required=True)

    args = parser.parse_args()

    image_pil, masks = get_segmented(args.image, args.class_name)

    process_image(image_pil, masks, args.output, args.x, args.y, args.class_name)



if __name__=='__main__':
    main()
