# ATUNDA - Afrogenic dance moves
This repository contains the dataset and source code used in our paper submitted to the ACM Journal of Computing and Cultural Heritage.

The dataset of Afrogenic dance moves has been systematically collected and curated for training machine learning models for dance move classification. The dataset includes nearly 400 move sequences covering 13 distinct dance moves from African and African Diaspora performative culture.

This dataset contains Afrogenic dance motion data extracted from the [Atunda videos](https://www.atunda.live/) using the MediaPipe pose detection library.
Each subfolder under `data/` corresponds to a dance move, and within each folder there are JSON files from several performances of this dance move in the following file name convension:

-   `<dance type>_normalized_<performance ID>.json` --- normalized landmark coordinates.
-   `<dance type>_world_<performance ID>.json` --- landmarks in world coordinates.

------------------------------------------------------------------------

## Dataset Structure

    data/
    ├── akwaaba/
    │   ├── akwaaba_normalized_1.json
    │   ├── akwaaba_world_1.json
        ├── akwaaba_normalized_2.json
        ├── akwaaba_world_2.json
        └── ...
    ├── alanta/
    │   ├── alanta_normalized_1.json
    │   ├── alanta_world_2.json
        └── ...
    └── ...

------------------------------------------------------------------------

## JSON Format

Each JSON file contains motion data for multiple sequences:

``` json
{
  "frames": [
    {
      "nose": {"x": 0.12, "y": -0.48, "z": -0.13, "visibility": 0.98},
      "left_eye_inner": {  },
      "left_eye": {  },
      "...": "...",
      "right_foot_index": {  }
    },
    
  ]
}
```

-   The "frames" array contains the list of frames in the motion sequence of this file.
-   Each frame contains 33 landmarks.
-   For each landmark:
    -   `x`, `y`, `z`: 3D coordinates (normalized or world, depending on
        the file)
    -   `visibility`: Confidence score (0.0 to 1.0)

------------------------------------------------------------------------

## Landmark Order

The dataset follows the MediaPipe Pose landmark definitions.\
The 33 landmarks are:

    ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
     "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
     "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
     "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
     "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
     "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
     "left_heel", "right_heel", "left_foot_index", "right_foot_index"]

------------------------------------------------------------------------

## How to Open the Dataset in Python

### Load a Single JSON File

``` python
import json

with open("data/akwaaba/akwaaba_normalized.json", "r") as f:
    data = json.load(f)

print(len(data["frames"]))  # number of frames in the sequence

first_frame = data["frames"][0]
print(first_frame["nose"])  # coordinates of the nose landmark
```

### Load Multiple Sequences from a Folder

``` python
import json
import glob

files = glob.glob("data/akwaaba/*.json")
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        print(f"{file} → {len(data['frames'])} frames")
```

------------------------------------------------------------------------

## License

The data provided under the [Atunda license](LICENSE). 
Please note that:
- The license expires in 1 year from the time you download the data.
- You must delete the dataset when your license expires.
- You can renew the license and re-download the dataset, as some of the data may have changed. (Why? Because performers may request to remove their data, or contribute new data). 
