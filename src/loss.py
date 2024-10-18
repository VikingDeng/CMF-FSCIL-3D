import torch
import torch.nn as nn
import torch.nn.functional as F

class LosswithIMG(nn.Module):
    def __init__(self,logger):
        super().__init__()
        self.labels = None
        self.main_labels = None
        self.all_labels = None
        self.batch_size = None
        self.logger = logger

    def compute_embeddings(self, pc_embed, text_embed, image_embed, logit_scale):
        logits_per_pc_text = logit_scale * pc_embed @ text_embed.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed.t()
        return logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc

    def forward(self, main_labels, all_labels, outputs):
        device = outputs['pc_embed'].device
        self.main_labels = main_labels.to(device)
        self.all_labels = all_labels.to(device)
        
        pc_embed = outputs['pc_embed'].to(device)
        text_embed = outputs['text_embed'].to(device)
        image_embed = outputs['image_embed'].to(device)
        logit_scale = outputs['logit_scale'].to(device)
        
        logits_per_pc_all = outputs['pc_all'].to(device)
        logits_per_pc_main = outputs['pc_main'].to(device)
        
        local_batch_size = pc_embed.size(0)
        self.batch_size = local_batch_size
        self.labels = torch.arange(self.batch_size).to(device)

        # Normalize embeddings
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        logits = self.compute_embeddings(pc_embed, text_embed, image_embed, logit_scale)

        logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc = logits

        loss = 0.3 * ((F.cross_entropy(logits_per_pc_text, self.labels) + F.cross_entropy(logits_per_text_pc, self.labels)) / 2 +
                    (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2) \
            + 0.1 * self.get_fake_loss(logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc, logits_per_pc_all.size(-1)) \
            + 0.4 * F.cross_entropy(logits_per_pc_all, self.all_labels) \
            + 0.2 * F.cross_entropy(logits_per_pc_main, self.main_labels)
        
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_all, dim=-1)
            correct = pred.eq(self.all_labels).sum()
            pc_all_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_main, dim=-1)
            correct = pred.eq(self.main_labels).sum()
            pc_main_acc = 100 * correct / local_batch_size

        # 检查是否存在异常值
        if not torch.isfinite(loss):
            self.logger.error(f"Detected non-finite loss value: {loss.item()}")
            self.logger.error(f"PC Embed: {pc_embed}")
            self.logger.error(f"Text Embed: {text_embed}")
            self.logger.error(f"Image Embed: {image_embed}")
            self.logger.error(f"Logits per PC-Text: {logits_per_pc_text}")
            self.logger.error(f"Logits per Text-PC: {logits_per_text_pc}")
            self.logger.error(f"Logits per PC-Image: {logits_per_pc_image}")
            self.logger.error(f"Logits per Image-PC: {logits_per_image_pc}")
            self.logger.error(f"Logits per PC-All: {logits_per_pc_all}")
            self.logger.error(f"Logits per PC-Main: {logits_per_pc_main}")
            raise ValueError("Non-finite loss detected")

        return {'loss': loss, 'pc_image_acc': pc_image_acc, 'pc_text_acc': pc_text_acc,
                'pc_all_acc': pc_all_acc, 'pc_main_acc': pc_main_acc}

    def get_fake_loss(self, logits_pc_text, logits_text_pc, logits_pc_image, logits_image_pc, n_classes):
        logits_per_pc_text, logits_per_text_pc, logits_per_pc_image, logits_per_image_pc = logits_pc_text.clone(), logits_text_pc.clone(), logits_pc_image.clone(), logits_image_pc.clone()
        mask = float('-inf')
        counts = int(self.batch_size / 5)

        for i in range(self.batch_size):
            if i % 5 == 0:
                continue
            logits_per_pc_text[i, i % n_classes] = mask
            logits_per_text_pc[i, i % n_classes] = mask
            logits_per_pc_image[i, i % n_classes] = mask
            logits_per_image_pc[i, i % n_classes] = mask

        new_labels = torch.arange(self.batch_size).to(logits_pc_text.device) % n_classes
        for category in range(counts):
            new_labels[category * 5:category * 5 + 5] = category * 5 % n_classes

        loss = (F.cross_entropy(logits_per_pc_text, new_labels) + F.cross_entropy(logits_per_text_pc, new_labels)) / 2 + \
               (F.cross_entropy(logits_per_pc_image, new_labels) + F.cross_entropy(logits_per_image_pc, new_labels)) / 2

        return loss

