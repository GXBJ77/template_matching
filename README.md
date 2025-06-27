# CAPTCHA Recognizer
A simple template-matching based CAPTCHA recognizer for 5-character captchas using OpenCV.
I used [Poetry](https://python-poetry.org/docs/) for package and environment management.

## 🧩 Task Description
The goal is to identify the unseen CAPTCHAs.

Characteristics of the Captchas:
- Always contain 5 characters (from A-Z and 0-9).
- Use the same font and spacing.
- Consistent background and foreground colors.
- No character skew or deformation.

(24 image and text pairs are given instead of 25.)

This makes it feasible to solve with template matching.

## 💡 Solution
### Step 1: Extract characters from given images as template 
check ```extract_characters```

For each labeled CAPTCHA image:

- Convert the image to **grayscale**.
- Separate the foreground and background using **Otsu's thresholding**.
- Detect **character contours** using `cv2.findContours`.
- Extract **bounding boxes** around each contour and sort them from **left to right**.
- **Crop and Resize (normalize)** each character image to a fixed size (default: **10×15 pixels**).
- Save each character image to build a template dataset.

### Step 2: Build the representative template for each character 
check ```load_templates```

All extracted characters are grouped by label (e.g. 'A', 'B', etc.).
Each group of character images is averaged to form a representative template per character.

### Step 3: Match Characters from Unseen Captchas 
check ```match_characters```
- Preprocess new CAPTCHA image (grayscale, thresholding, crop, resize and sort). Same as **Step 1**. check ```preprocess```
- For each detected region:
  - Match against all templates using ```cv2.matchTemplate``` with method ```cv2.TM_CCOEFF_NORMED```.
  - Choose the best match as the predicted character.

### Step 4: Save
check ```__call___```

Given the input image path (accept both png and txt format), save to output folder. 


# 👨‍💻 How to run the code
### 📂 Folder Structure:
```
project/
├── captcha.py                 # Full script
├── sampleCaptchas/            # Image folder provided
│   └── input/                 # Input CAPTCHA images as provided
│   └── output/                # txt label of CAPTCHA images as provided
│   └── template/              # template images extracted
│   └── unseen/                # folder created to save the unseen image predicted results
├── text_captcha.py            # unittest
├── poetry.lock                # poetry packages                 
├── pyproject.toml             # poetry packages                                 
```

### 📦 Install Poetry
If you don’t have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then install the dependencies:
```bash
poetry install --no-root
```

### run the script
Use the following command from the project root:

```bash
poetry run python3 captcha.py --test_image ./sampleCaptchas/input/input21.jpg --test_output ./sampleCaptchas/unseen/output21.txt
```
or 
```bash
poetry run python3 captcha.py --test_image ./sampleCaptchas/input/input21.txt --test_output ./sampleCaptchas/unseen/output21.txt
```

I also write unittest cases to check the code work as expected: 
check ```test_captcha.py```

```bash 
poetry run python3 -m unittest test_captcha.py
```
