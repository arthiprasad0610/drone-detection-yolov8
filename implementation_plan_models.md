# Implementation Plan - Model Versions & Custom Coloring

This plan outlines the changes needed to support multiple model versions and implement specific coloring for detected flying objects in `live_server_advanced.py`.

## 1. Model Version Discovery & Management
- **Scan for Models**: Create a utility function to find all `.pt` files in `captured_datasets/training_results/` and the base `yolov8n.pt`.
- **API Endpoints**:
    - `GET /list-models`: Returns available model versions.
    - `POST /switch-model`: Switches the active model in `FrameManager`.
- **FrameManager Update**: Add `switch_model` method to reload YOLO/SAHI with a different weights file.

## 2. Detection & Visualization Enhancements
- **Custom Coloring**: Update `draw_flying_objects_on_frame` to use specific BGR colors:
    - **Drone**: Red `(0, 0, 255)`
    - **Airplane**: Brown `(19, 69, 139)`
    - **Helicopter**: Pink `(203, 192, 255)`
    - **Bird**: Yellow `(0, 255, 255)`
- **Dynamic Labels**: Ensure the labels match the selected model's class names or use the custom mapping if it's a known class.

## 3. Web Interface (Dashboard)
- **Model Selector**: Add a premium-styled dropdown menu in the "System Information" or a new "Model Settings" section.
- **Interactive Switching**: Implement Javascript to:
    - Load available models on page load.
    - Call `/switch-model` when a user selects a new version.
    - Show a success/loading notification.

## 4. Metadata Persistence
- Ensure `.txt` file generation in `capture_detected_frame_with_labels` remains consistent with the selected model's output.

---
**Colors Reference (BGR for OpenCV):**
- Drone: `(0, 0, 255)` (Red)
- Airplane: `(19, 69, 139)` (Brown)
- Helicopter: `(203, 192, 255)` (Pink)
- Bird: `(0, 255, 255)` (Yellow)
