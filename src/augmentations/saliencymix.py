import numpy as np
import torch
import cv2

class SaliencyMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def saliency_bbox(self, img, lam):
        """
        Find a saliency-guided bounding box (y/x order respected for CHW tensors).
        """
        _, H, W = img.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (_, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        cy, cx = np.unravel_index(np.argmax(saliencyMap), saliencyMap.shape)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        return y1, y2, x1, x2

    def __call__(self, images, labels):
        r = np.random.rand(1)
        if self.beta > 0 and r < self.prob:
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(images.size(0), device=images.device)
            labels_a, labels_b = labels, labels[rand_index]

            y1, y2, x1, x2 = self.saliency_bbox(images[rand_index[0]], lam)
            images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
            lam = 1 - ((y2 - y1) * (x2 - x1) / (images.size(-1) * images.size(-2)))
            return images, labels_a, labels_b, lam
        else:
            return images, labels, labels, 1.0
        