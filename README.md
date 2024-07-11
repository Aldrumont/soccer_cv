
# Soccer CV Project

This project applies computer vision and artificial intelligence to analyze soccer videos. Using YOLO models for player and ball detection, Norfair for movement tracking, and OpenCV for image processing, it identifies teams, determines ball possession, and tracks player movements.

For this project, a segment from the 2023/2024 Champions League final broadcast available [here](https://www.youtube.com/watch?v=cD537FR7Awc) was used, and the video was downloaded in 1080p resolution.

## Scripts Overview

### `img_hsv_extract.py`

This script extracts HSV components from an image. It allows users to use a trackbar to determine the appropriate HSV values to filter specific colors for teams, goalkeepers, referees, etc.

#### Usage
1. Load an image.
2. Adjust the trackbars to find the HSV range that isolates the desired color.
3. Use these HSV values to set up the main script.

For example, to isolate the uniform color of Borussia Dortmund, the selected HSV range might be:
```python
# "Borussia": [[H_min, H_max], [S_min, S_max], [V_min, V_max]],
"Borussia": [[35, 42], [74, 100], [40, 100]]
```
Video demonstration for determining the HSV range of Borussia Dortmund uniform can be found here:

readme_utils/hsv_extract_video.mp4

### `soccer_video_analysis.py`

This is the main script that performs the video analysis. It has been tested with YOLOv5 and YOLOv8. To use other YOLO versions, modify the model name and the `yolo_version` variable.

#### Configuration

1. **Select YOLO Version:**
   Set the `yolo_version` variable to the desired YOLO version (5 or 8).

2. **Set HSV Values:**
   Populate the `class_main_hsv_color` and `class_hsv_range` variables.
   - `class_main_hsv_color`: Set the main HSV color for referees, goalkeepers, teams, etc.
   - `class_hsv_range`: Use `img_hsv_extract.py` or an online color picker to find the HSV range that isolates the desired colors.

Example values for `class_hsv_range`:
```python
class_hsv_range = {
    # "Borussia": [[H_low, H_high], [S_low, S_high], [V_low, V_high]],
    "Borussia": [[31, 48], [50, 100], [70, 100]],
    "Real Madrid": [[0, 360], [0, 15], [85, 100]],
    "Borrusia_GK": [[0, 10], [70, 94], [84, 100]],
    "Referee": [[119, 149], [50, 100], [50, 100]],
    "Real Madrid_GK": [[54, 66], [61, 100], [61, 100]],
}
```

Example values for `class_main_hsv_color`:
```python
class_main_hsv_color = {
    "Borussia": (51, 87, 96),
    "Real Madrid": (220, 0, 100),
    "Referee": (195, 70, 88),
    "Borrusia_GK": (12, 80, 98),
    "Real Madrid_GK": (78, 80, 87),
}
```

3. **Set Video Parameters:**
   Update the following lines with your desired parameters:
   ```python
   video_path = 'final_champions.mp4'
   output_path = 'processed_video.mp4'
   read_frames_from_video(video_path, output_path, show_video=True, save_video=True, skip_frames=1, start_time=60*14, end_time=60*16)
   ```

#### Parameters Explanation

- `video_path`: Path to the input video file.
- `output_path`: Path to save the processed video.
- `show_video`: Boolean to display the video while processing.
- `save_video`: Boolean to save the processed video.
- `skip_frames`: Number of frames to skip between each processing step to speed up the processing.
- `start_time`: Start time in seconds from where to begin processing the video.
- `end_time`: End time in seconds to stop processing the video.

By configuring these parameters, you can customize the analysis to fit your needs and extract valuable insights from soccer videos.

## Installation and Requirements

To install the necessary dependencies, create a virtual environment and install the required packages using `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Project Inspiration

This project was inspired by the Tryolabs project [Measuring Soccer Ball Possession: AI Video Analytics](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics).

## YOLO Models

YOLO already has well-trained classes, for example, it performs well in detecting people. However, detecting the ball is not as effective. In this repository, two models were used for ball detection:
- The YOLOv5 model was obtained from the Tryolabs project mentioned above.
- The YOLOv8 model was obtained from the project [YOLOv8-football](https://github.com/noorkhokhar99/YOLOv8-football?tab=readme-ov-file).

## Possible Improvements

1. **Ball Possession Determination:** Currently, ball possession is determined by calculating which detected player's bounding box is closest to the ball's centroid. This method may cause errors depending on the camera angle and player positions in the video.
2. **Ball Detection:** The ball detection models can still be improved. There are many frames where the ball is not detected.

## Potential Applications

- **Tactical Analysis:** Providing detailed insights into player movements and strategies.
- **Training and Development:** Helping coaches and players identify areas for improvement.
- **Fan Engagement:** Offering advanced analysis and statistics during live broadcasts.

## Future Work

- **2D Radar:** Create a 2D radar displaying player positions.
- **Ball Possession Tracking:** Track and count ball possession for each team.
- **Statistics Generation:** Develop statistics for performance analysis.
- **Semi-Automatic offside detection:** Implement a semi-automatic offside detection system.

