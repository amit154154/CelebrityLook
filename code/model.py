import pytorch_lightning as pl
from torch import nn
import torch
import torch.nn.functional as F

class mapper_train(pl.LightningModule):
    def __init__(self, decoder, mapping, mean_latent_clip, mean_w, c):
        super().__init__()
        self.decoder = decoder
        self.mapping = mapping
        self.decoder.eval()
        self.mapping.eval()
        self.clip_func = c
        self.mapper = nn.Sequential(
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 768), nn.GELU(),
            nn.Linear(768, 23 * 512), nn.GELU()

        )
        self.mean_latent_clip = mean_latent_clip.cuda()
        self.mean_w = mean_w.reshape(1, 23, 512).cuda()

    def configure_optimizers(self):
        # params = list(self.mapper.parameters()) + list(self.clip_func.encoder.parameters(),)
        parms = list(self.mapper.parameters())
        return torch.optim.Adam(parms, lr=3e-5)

    def training_step(self, batch, batch_idx):
        random_latents = batch.cuda()
        batch_size = random_latents.shape[0]
        random_latents = torch.randn(random_latents.shape).cuda()

        z_plus_random = random_latents.reshape(batch_size, 512)
        w_space_random = self.mapping(z_plus_random).reshape(batch_size, 1, 512)
        w_space_random = w_space_random.repeat(1, 23, 1)

        random_images = self.decoder(w_space_random)

        preprocess_images = self.clip_func.process_images(random_images)
        random_images_features_clip = self.clip_func.image_features_clip(preprocess_images).cuda()
        # random_images_features_distil = self.clip_func.image_features_distil(preprocess_images).cuda()

        w_spaces = []
        for i in range(batch_size):
            w_plus = self.mapper(
                (random_images_features_clip[i].reshape(1, 768) - self.mean_latent_clip).float()).reshape(1, 23, 512)
            w_space = self.mean_w + w_plus
            w_spaces.append(w_space)

        if batch_idx % 200 == 0:
            generated_image = self.decoder(w_space)
            self.logger.log_image(key="samples", images=[random_images[-1], generated_image])

        w_spaces = torch.stack(w_spaces).reshape(batch_size, 23, 512)

        clip_encodeing_mse = F.mse_loss(w_spaces, w_space_random, reduction="mean").cuda().mean()
        cosin_similarty = 1 - F.cosine_similarity(w_spaces, w_space_random).abs().mean()

        total_loss = 5 * clip_encodeing_mse

        if batch_idx % 10 == 0:
            self.log_dict({'loss': total_loss, 'mse': clip_encodeing_mse, 'cosin_similarty': cosin_similarty})

        return total_loss

    def forward(self, clip_features):
        with torch.no_grad():
            w_plus = self.mapper(
                (clip_features.reshape(clip_features.shape[0], 768) - self.mean_latent_clip).float()).reshape(
                clip_features.shape[0], 23, 512)
            w_space = self.mean_w + w_plus
            generated_image = self.decoder(w_space)
        return generated_image

    def inversion(self, image):
        with torch.no_grad():
            preprocess_images = self.clip_func.process_images(image)
            clip_features = self.clip_func.image_features_clip(preprocess_images).cuda()
            w_plus = self.mapper(
                (clip_features.reshape(clip_features.shape[0], 768) - self.mean_latent_clip).float()).reshape(
                clip_features.shape[0], 23, 512)
            w_space = self.mean_w + w_plus
            # generated_image = self.decoder(w_space)
        return w_space

    def get_text_delta(self, features_target, features_src):
        self.eval()
        with torch.no_grad():
            text_features = features_src
            text_feactures_target = features_target
            deltaW_src = self.mapper(text_features)
            deltaW_target = self.mapper(text_feactures_target)
            deltaW = (deltaW_target - deltaW_src).reshape(1, 23, 512)
        return deltaW



