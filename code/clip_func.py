import clip
import torch
from torchvision import transforms, utils


class Clip_Moudle:
    def __init__(self):
        self.clip_model ,self.preprocess_clip = clip.load('ViT-L/14' ,device='cuda')
        # self.clip_distil = torch.jit.load('../input/clip-distl/clip_jit_1-2.pt',map_location = 'cuda')
        self.clip_model.eval()
        # self.clip_distil.eval()

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.inv_normalize = transforms.Normalize(
            mean= [- m /s for m, s in zip(mean, std)],
            std= [ 1 /s for s in std]
        )

    def image_features_clip(self ,preproces_image):
        with torch.no_grad():
            encode =  self.clip_model.encode_image(preproces_image.cuda())
            preproces_image = preproces_image.cpu()
            return encode
    def image_features_distil(self ,preproces_image):
        with torch.no_grad():
            encode =  self.clip_distil(preproces_image.cuda())
            preproces_image = preproces_image.cpu()
            return encode

    def image_similarty(self ,image_features_1 ,image_features_2):
        image_features_1 /= image_features_1.norm(dim=-1, keepdim=True)
        image_features_2 /= image_features_2.norm(dim=-1, keepdim=True)
        similarity = image_features_2.cpu().detach().numpy() @ image_features_1.cpu().detach().numpy().T
        image_features_1 = image_features_1.cpu()
        image_features_2 = image_features_2.cpu()
        return torch.tensor( 1 -similarity ,requires_grad=True)

    def process_images(self ,generated_images):
        process_images = []
        transform = transforms.ToPILImage()
        for image in generated_images:
            process_images.append(self.preprocess_clip(transform(image)))
        return torch.stack(process_images)
