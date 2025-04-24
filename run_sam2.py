import torch
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# For detecting humans 
import torchvision
from torchvision.transforms import functional as F
import cv2

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device = 'cuda:0')


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# Load a pretrained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to('cpu')
model.eval()


video_dir =  'C:/Users/hkwon13/OneDrive - Brown University/Desktop/KiteVideoFrames'

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
# frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)
prompts = {}  # hold all the clicks we add for visualization


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

total_humans = 0
for frameIdx in range(len(frame_names)):
    frame = cv2.imread(video_dir + "/" + frame_names[frameIdx])
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).to('cpu')
    outputs ='none'
    with torch.no_grad():
        outputs = model([image_tensor])
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    human_coordinates = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if label == 1 and score > 0.8:  # Class 1 corresponds to 'person' in COCO
            human_coordinates.append(box)
            total_humans += 1


            bounding_box = np.array(box, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )

            # show the results on the current (interacted) frame
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
            show_box(bounding_box, plt.gca())
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

print(total_humans)



    # Let's add a positive click at (x, y) = (200, 300) to get started on the first object
    # points = np.array([[200, 300]], dtype=np.float32)
    # # for labels, `1` means positive click and `0` means negative click
    # labels = np.array([1], np.int32)
    # prompts[ann_obj_id] = points, labels
    # _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    #     inference_state=inference_state,
    #     frame_idx=ann_frame_idx,
    #     obj_id=ann_obj_id,
    #     points=points,
    #     labels=labels,
    # )

# run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }


# show the results on the current (interacted) frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {ann_frame_idx}")
# plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
# for i, out_obj_id in enumerate(out_obj_ids):
#     show_points(*prompts[out_obj_id], plt.gca())
#     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)



# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     state = predictor.init_state(video_dir)

#     # add new prompts and instantly get the output on the same frame
#     frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

#     # propagate the prompts to get masklets throughout the video
#     for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
#         ...