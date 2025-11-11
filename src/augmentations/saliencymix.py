import numpy as np
import torch
import cv2

class SaliencyMix:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def saliency_bbox(self, img, lam):
        size = img.size()
        W, H = size[1], size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

        temp_img = img.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (_, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        x, y = np.unravel_index(np.argmax(saliencyMap), saliencyMap.shape)
        bbx1, bby1 = np.clip(x - cut_w // 2, 0, W), np.clip(y - cut_h // 2, 0, H)
        bbx2, bby2 = np.clip(x + cut_w // 2, 0, W), np.clip(y + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, images, labels):
        r = np.random.rand(1)
        if self.beta > 0 and r < self.prob:
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(images.size(0)).cuda()
            labels_a, labels_b = labels, labels[rand_index]

            bbx1, bby1, bbx2, bby2 = self.saliency_bbox(images[rand_index[0]], lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
            return images, labels_a, labels_b, lam
        else:
            return images, labels, labels, 1.0
