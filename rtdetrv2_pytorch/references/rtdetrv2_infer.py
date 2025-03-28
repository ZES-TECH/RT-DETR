import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw

import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig

class Inferencer(nn.Module):
    """
    A class that holds:
      - The underlying detection model + postprocessor
      - The required transforms
      - A method for inference on PIL Images
      - A method for drawing bounding boxes on PIL Images
    """
    def __init__(self, cfg, device='cpu', score_thr=0.001):
        super().__init__()
        # Wrap final model
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.device = device
        self.score_thr = score_thr
        
        # Precompute transformations (to avoid re-initializing every time)
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

        self.class_names = { 
            0: 'person', 
            1: 'pressmachine', 
            2: 'robotarm', 
            3: 'fence', 
            4: 'forklift', 
            5: 'cart', 
            6: 'lot', 
            7: 'fire', 
            8: 'smoke'
        }
        
        self.color_map = {
            0: '#1f77b4', 
            1: '#ff7f0e', 
            2: '#2ca02c', 
            3: '#d62728', 
            4: '#9467bd', 
            5: '#8c564b', 
            6: '#e377c2', 
            7: '#7f7f7f', 
            8: '#bcbd22'
        }
        
    def forward(self, images, orig_sizes):
        """
        Overriding nn.Module's forward to handle our detection + postprocessing.
        images: Tensor(B, C, H, W)
        orig_sizes: shape (B, 2) for the original (width, height)
        """
        outputs = self.model(images)  # raw outputs
        outputs = self.postprocessor(outputs, orig_sizes)
        return outputs  # (labels, boxes, scores)
    
    @torch.no_grad()
    def process_pil_image(self, im_pil):
        """
        Takes a single PIL image, transforms and runs inference,
        then returns an annotated PIL image.
        """
        # Convert to device
        im_data = self.transforms(im_pil)[None].to(self.device)
        
        labels, boxes, scores = self(im_data, self.orig_size)
        
        # Draw bounding boxes on a copy
        out_image = im_pil.copy()
        self.draw_on_image(out_image, labels, boxes, scores)
        
        return out_image
    
    def draw_on_image(
        self, 
        pil_img, 
        batch_labels, 
        batch_boxes, 
        batch_scores,
    ):
        """
        Draw bounding boxes & labels on a single PIL image.
        The text is displayed inside a filled box *above* the bounding box.

        - Bounding boxes are colored based on class.
        - The class name and confidence score are displayed inside a colored box above.
        """

        draw_handle = ImageDraw.Draw(pil_img)
        labels = batch_labels[0]
        boxes = batch_boxes[0]
        scores = batch_scores[0]
        
        # Filter by confidence threshold
        mask = scores > self.score_thr
        labels = labels[mask]
        boxes = boxes[mask]
        scores = scores[mask]

        for label_id, box, scr in zip(labels, boxes, scores):
            label_id = int(label_id.item())
            score_val = scr.item()

            # Convert label ID -> class name (or fallback)
            class_name = self.class_names.get(label_id, f"cls_{label_id}")

            # Pick bounding box color based on class name
            color = self.color_map.get(label_id, "red")

            # Draw bounding box
            draw_handle.rectangle(list(box), outline=color, width=3)

            # Create text label
            text_str = f"id: NaN {class_name} {round(score_val, 2)}"

            # Compute text size (Pillow 8.0+ supports textbbox, else use textsize)
            text_size = draw_handle.textbbox((0, 0), text_str)  # (x0, y0, x1, y1)
            text_w = text_size[2] - text_size[0]
            text_h = text_size[3] - text_size[1]

            # Calculate text position (above the bounding box)
            text_x = box[0]
            text_y = max(box[1] - text_h - 5, 0)  # Prevent text from going out of bounds

            # Draw filled rectangle for text background
            draw_handle.rectangle(
                [text_x, text_y, text_x + text_w + 4, text_y + text_h + 2],
                fill=color
            )

            # Draw text on top of the rectangle
            draw_handle.text((text_x + 2, text_y), text_str, fill="white")

    def set_original_img_size(self, w, h):
        self.orig_size = torch.tensor([w, h])[None].to(self.device)

def main(args):
    """
    - If args.im_file is a video -> process it, write out a single annotated video.
    - If args.im_file is a folder -> load all images, run inference, write out a single video.
    - Otherwise, treat as a single image and save results to a single .jpg
    """
    # Setup config
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Must provide --resume to load model state.')

    # Load state dict
    cfg.model.load_state_dict(state)

    # Create the Inferencer class object
    inferencer = Inferencer(cfg, device=args.device, score_thr=0.6).to(args.device)
    inferencer.eval()

    path_ = args.im_file
    ext = os.path.splitext(path_)[1].lower()

    # --- CASE 1: Input is a Video ---
    if ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        cap = cv2.VideoCapture(path_)
        if not cap.isOpened():
            print(f"Error opening video: {path_}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set original size to avoid recomputing always
        inferencer.set_original_img_size(width, height)
        
        out_name = f'{args.output}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_name, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Convert to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_rgb)
            # If this image is different size, resize to match the first
            if im_pil.size != (width, height):
                im_pil = im_pil.resize((width, height))
            
            # Run inference
            out_pil = inferencer.process_pil_image(im_pil)
            
            # Convert back to BGR for saving
            out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            out_writer.write(out_bgr)
            print(f"Processed frame {frame_count}")

        cap.release()
        out_writer.release()
        print(f"Saved annotated video as {out_name}")

    # --- CASE 2: Input is a folder of images -> one output video
    elif os.path.isdir(path_):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        files = sorted([
            f for f in os.listdir(path_)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        if not files:
            print(f"No images in directory: {path_}")
            return
        
        # We'll read the first image to set up the video size
        first_img_path = os.path.join(path_, files[0])
        test_pil = Image.open(first_img_path).convert('RGB')
        w0, h0 = test_pil.size
        
        # Output video
        out_video_name = f'{args.output}.mp4'
        fps = 30.0  # arbitrary
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_name, fourcc, fps, (w0, h0))

        # Set original size to avoid recomputing always
        inferencer.set_original_img_size(w0, h0)

        for idx, fname in enumerate(files):
            fpath = os.path.join(path_, fname)
            im_pil = Image.open(fpath).convert('RGB')
            # If this image is different size, resize to match the first
            if im_pil.size != (w0, h0):
                im_pil = im_pil.resize((w0, h0))
            
            out_pil = inferencer.process_pil_image(im_pil)
            frame_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)
            out_writer.write(frame_bgr)
            print(f"Processed {idx+1}/{len(files)}: {fname}")

        out_writer.release()
        print(f"Saved folder sequence results as {out_video_name}")

    # --- CASE 3: Single Image
    else:
        if not os.path.isfile(path_):
            print(f"File not found: {path_}")
            return
        
        im_pil = Image.open(path_).convert('RGB')

        # Set original size to avoid recomputing always
        inferencer.set_original_img_size(im_pil.size)

        out_pil = inferencer.process_pil_image(im_pil)
        
        save_path = f'{args.output}.jpg'
        out_pil.save(save_path)
        print(f"Saved single-image result as {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-f', '--im-file', type=str, required=True,
                        help="Image, folder of images, or video")
    parser.add_argument('-o', '--output', type=str, default='result', help='name of output file (without extension)')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
