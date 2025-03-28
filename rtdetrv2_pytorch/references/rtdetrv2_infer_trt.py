import contextlib
import collections
from collections import OrderedDict
from packaging import version
import time
import torch
import numpy as np
import os
import cv2
import PIL
from PIL import Image, ImageDraw

import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import tensorrt as trt

class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', max_batch_size=32, img_size=(640, 640), score_thr=0.6, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.max_batch_size = max_batch_size
        self.size = img_size
        self.score_thr = score_thr
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        self.time_profile = TimeProfiler()

        # For Visualization
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

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build binddings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)
            
            assert self.bindings[n].data.dtype == blob[n].dtype, '{} dtype mismatch'.format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    
    def __call__(self, img_pil):
        im_data = self.preprocess(img_pil)
        blob = {
        'images': im_data.to(self.device), 
        'orig_target_sizes': self.orig_size.to(self.device),
        }
        output = self.run_torch(blob)

        speed = self.speed(blob, 10)
        print(f"Speed: {speed:.4f} s")
        return output

    def set_original_img_size(self, w, h):
        self.orig_size = torch.tensor([w, h])[None].to(self.bindings['orig_target_sizes'].data.dtype)

    def synchronize(self, ):
        torch.cuda.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self.run_torch(blob)

        return self.time_profile.total / n 


    @staticmethod
    def onnx2tensorrt():
        pass

    def preprocess(self, pil_image):
        """
        Resize a PIL image and convert to a NumPy float32 tensor 
        with shape (C, H, W), range [0, 1].
        """

        # 1) Resize using PIL with the exact interpolation that torchvision uses
        resized_img = pil_image.resize(self.size, Image.BILINEAR)  # Antialiasing enabled

        # 2) Convert to a NumPy array
        arr = np.asarray(resized_img, dtype=np.float32)  # Ensures identical dtype as ToTensor()

        # 3) Scale from [0, 255] -> [0, 1] exactly like ToTensor()
        arr *= (1.0 / 255.0)

        # 4) Convert (H, W, C) -> (C, H, W) to match PyTorch's layout
        arr = np.transpose(arr, (2, 0, 1))

        # 5) Convert to a contiguous torch tensor (ensures same memory layout as ToTensor)
        tensor = torch.from_numpy(arr).contiguous()

        return tensor[None]  # Output: shape (3, H, W), dtype=torch.float32

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
            if version.parse(PIL.__version__) >= version.parse("8.0.0"):
                text_size = draw_handle.textbbox((0, 0), text_str)  # (x0, y0, x1, y1)
                text_w = text_size[2] - text_size[0]
                text_h = text_size[3] - text_size[1]
            else:
                text_w, text_h = draw_handle.textsize(text_str)

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

def main(args):
    """
    - If args.im_file is a video -> process it, write out a single annotated video.
    - If args.im_file is a folder -> load all images, run inference, write out a single video.
    - Otherwise, treat as a single image and save results to a single .jpg
    """

    # Create the Inferencer class object
    inferencer = TRTInference(args.trt_file, device=args.device, max_batch_size=1, img_size=(640, 640), score_thr=args.conf_score, verbose=True)

    path_ = args.im_file
    ext = os.path.splitext(path_)[1].lower()
    file_name = os.path.splitext(path_)[0]
    out_file_name = f'{file_name}_trt_output'

    # --- CASE 1: Input is a Video ---
    if ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        cap = cv2.VideoCapture(path_)
        if not cap.isOpened():
            print(f"Error opening video: {path_}")
            return
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        inferencer.set_original_img_size(orig_width, orig_height)
        
        out_name = f'{out_file_name}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_name, fourcc, fps, (orig_width, orig_height))
        
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
            if im_pil.size != (orig_width, orig_height):
                im_pil = im_pil.resize((orig_width, orig_height))
            
            # Run inference
            output = inferencer(im_pil)

            # Draw
            out_pil = im_pil.copy()
            inferencer.draw_on_image(out_pil, output['labels'], output['boxes'], output['scores'])
            
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
        orig_width, orig_height = test_pil.size
        
        # Output video
        out_video_name = f'{out_file_name}.mp4'
        fps = 30.0  # arbitrary
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_video_name, fourcc, fps, (orig_width, orig_height))

        # Set original size to avoid recomputing always
        inferencer.set_original_img_size(orig_width, orig_height)

        for idx, fname in enumerate(files):
            fpath = os.path.join(path_, fname)
            im_pil = Image.open(fpath).convert('RGB')
            # If this image is different size, resize to match the first
            if im_pil.size != (orig_width, orig_height):
                im_pil = im_pil.resize((orig_width, orig_height))
            
            # Perform inference
            output = inferencer(im_pil)

            # Draw
            out_pil = im_pil.copy()
            inferencer.draw_on_image(out_pil, output['labels'], output['boxes'], output['scores'])
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
        inferencer.set_original_img_size(*im_pil.size)

        # Perform inference
        output = inferencer(im_pil)

        # Draw
        out_pil = im_pil.copy()
        inferencer.draw_on_image(out_pil, output['labels'], output['boxes'], output['scores'])
        
        save_path = f'{out_file_name}.jpg'
        out_pil.save(save_path)
        print(f"Saved single-image result as {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, required=True)
    parser.add_argument('-f', '--im-file', type=str, required=True,
                        help="Image, folder of images, or video")
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-s', '--conf_score', type=float, default=0.6, help='Confidence score thresh')
    args = parser.parse_args()
    main(args)
