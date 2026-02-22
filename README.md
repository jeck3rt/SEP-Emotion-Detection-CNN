Branch Changes — chih-chi/test
Summary of all improvements made on this branch.

New File: utils.py
Extracted shared boilerplate that was previously duplicated across demo.py, videodemo.py, and ImageExampleScripts.py into a single shared module.

Contains:

emotion_labels — the 6 emotion class names
device — CUDA/CPU device selection
current_folder — resolved path to the project root
face_classifier — OpenCV Haar cascade face detector
Facefinder — PIL transform that auto-crops the first detected face (used during dataset evaluation)
img_transforms — standard inference transform pipeline (grayscale → resize → tensor → normalize)
eval_transforms — evaluation transform pipeline with Facefinder pre-crop included
load_model() — loads CNN from weights.pth, moves to device, sets eval mode
draw_emotion_bars() — draws a semi-transparent probability bar panel on a frame
evaluation.py
Bug fix — FERPlus 7-class mismatch: FERPlus contains a neutral class that the model was not trained on. When ImageFolder loaded the test set, alphabetical ordering caused label indices to shift, resulting in IndexError: Target 6 is out of bounds. Added a post-load filtering step that removes unknown classes and remaps remaining labels to match the model's 6-class output.

New visualizations: After evaluation completes, two figures are now generated and saved automatically.

evaluation_stats.png:

Left: confusion matrix heatmap with per-cell counts
Right: per-class accuracy bar chart (blue = above overall average, orange = below)
evaluation_samples.png:

6 × 5 grid of actual grayscale face images from the test set
Each row corresponds to one true emotion class
Each cell title shows the predicted label in green (correct) or red (wrong)
Refactored to import shared utilities from utils.py.

demo.py
Performance — GradCAM initialization moved outside the webcam loop: Previously GradCAM(model=model, target_layers=...) was called every frame, recreating the object on each inference. It is now initialized once before the loop.

Visualization improvements:

Removed redundant class index number from the on-screen label (was "3 happy", now "happy  87%")
Added confidence percentage next to the predicted emotion label
Added FPS counter in the top-left corner (yellow text)
Window close — X button now works: Added cv2.getWindowProperty check so clicking the window's close button exits the loop cleanly and releases the webcam, in addition to the existing q key shortcut.

Refactored to import shared utilities from utils.py.

videodemo.py
Performance — GradCAM initialization moved outside the video loop (same fix as demo.py).

Visualization improvements:

Added confidence percentage to the emotion label overlay
Added draw_emotion_bars probability panel to output video frames (was missing, only demo.py had it)
Refactored to import shared utilities from utils.py.

ImageExampleScripts.py
Previously the saved output image only contained the Grad-CAM heatmap with no annotation.

Added to the output image:

Emotion label + confidence percentage overlaid above the face bounding box
Probability bar panel in the bottom-right corner
Output is now consistent with the live demo display in demo.py.

Refactored to import shared utilities from utils.py.

Training.py
Fixed deprecated PyTorch mixed-precision API:

Before	After
torch.amp.GradScaler()	torch.amp.GradScaler(amp_device)
torch.amp.autocast(device)	torch.amp.autocast(amp_device)
Passing the device type string explicitly silences the deprecation warning in newer PyTorch versions.

# SEP-Emotion-Detection-CNN

Dataset used for training: FERPlus -> https://www.kaggle.com/datasets/arnabkumarroy02/ferplus
needs PIL, CV2 Pytorch, numpy and GRAD-CAM
