import cv2
import numpy as np
import os
from glob import glob

class Captcha(object):
    def __init__(self, template_dir="./sampleCaptchas/templates/", 
                 input_folder="./sampleCaptchas/input/", 
                 input_format="jpg", 
                 output_folder="./sampleCaptchas/output/", 
                 output_format="txt", 
                 template_size=(10, 15),
                 ):
        os.makedirs(os.path.dirname(template_dir), exist_ok=True)
        self.template_dir = template_dir
        self.template_size = template_size
        self.input_folder = input_folder
        self.input_format = input_format
        self.output_folder = output_folder
        self.output_format = output_format

        images = sorted(glob(os.path.join(self.input_folder, f"*.{self.input_format}")))
        labels = sorted(glob(os.path.join(self.output_folder, f"*.{self.output_format}")))

        # only keep the pairs that both images and labels exist
        def extract_suffix(filename, prefix):
            base = os.path.splitext(os.path.basename(filename))[0]
            return base.replace(prefix, "")

        image_dict = {extract_suffix(f, "input"): f for f in images}
        label_dict = {extract_suffix(f, "output"): f for f in labels}

        # Step 3: find common suffixes
        common_suffixes = sorted(set(image_dict) & set(label_dict))

        # Step 4: keep only matched pairs
        self.matched_images = [image_dict[suffix] for suffix in common_suffixes]
        self.matched_labels = [label_dict[suffix] for suffix in common_suffixes]

        self.extract_characters()
        self.templates = self.load_templates()


    def preprocess(self, image_path, input_format):
        if input_format == "jpg":
            image = cv2.imread(image_path)
        elif input_format == "txt":
            with open(image_path, 'r') as f:
                height, width = map(int, f.readline().strip().split())

                pixels = []
                for line in f:
                    line = line.strip()
                    if line:
                        pixel_strings = line.split()
                        for pixel_str in pixel_strings:
                            rgb = list(map(int, pixel_str.split(',')))
                            pixels.append(rgb)

                # Convert and reshape to (height, width, 3)
                image = np.array(pixels, dtype=np.uint8).reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = sorted(boxes, key=lambda b: b[0])
        return gray, boxes



    def extract_characters(self):
        label_count = {}

        for image_path, label_path in zip(self.matched_images, self.matched_labels):
            with open(label_path, 'r') as f:
                label = f.read().strip()

            gray, boxes = self.preprocess(image_path, self.input_format)

            if len(boxes) != len(label):
                print(f"⚠️ Skipping {image_path}: expected {len(label)} chars, found {len(boxes)}")
                continue

            for i, (x, y, w, h) in enumerate(boxes):
                char_img = gray[y:y + h, x:x + w]
                char_img = cv2.resize(char_img, self.template_size)
                char = label[i]
                label_count[char] = label_count.get(char, 0) + 1
                out_path = os.path.join(self.template_dir, f"{char}_{label_count[char]}.png")
                cv2.imwrite(out_path, char_img)

    def load_templates(self):
        templates = {}

        for filepath in glob(os.path.join(self.template_dir, "*.png")):
            filename = os.path.basename(filepath)
            char = filename.split("_")[0]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if char not in templates:
                templates[char] = []
            templates[char].append(img)

        for char in templates:
            templ = np.mean(templates[char], axis=0)
            templates[char] = templ.astype(np.uint8)

        return templates
    
    def match_characters(self, image_path, method=cv2.TM_CCOEFF_NORMED):
        input_format = image_path.split('.')[-1]
        gray, boxes = self.preprocess(image_path, input_format)
        result = []

        for (x, y, w, h) in boxes:
            char_img = gray[y:y + h, x:x + w]
            char_img = cv2.resize(char_img, self.template_size)

            best_score = -np.inf
            best_char = None

            for char, templ in self.templates.items():
                res = cv2.matchTemplate(char_img, templ, method)
                score = res[0][0]

                if score > best_score:
                    best_score = score
                    best_char = char

            result.append(best_char)

        return ''.join(result)

    def __call__(self, im_path, save_path=None):
        predicted_label = self.match_characters(im_path)
        print(f"Predicted: {predicted_label}")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                f.write(predicted_label+'\n')
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Captcha recognizer")

    parser.add_argument("--template_dir", type=str, default="./sampleCaptchas/templates/", help="Directory to save or load templates")
    parser.add_argument("--input_folder", type=str, default="./sampleCaptchas/input/", help="Directory containing input images")
    parser.add_argument("--output_folder", type=str, default="./sampleCaptchas/output/", help="Directory containing output labels for training")
    parser.add_argument("--unseen_folder", type=str, default="./sampleCaptchas/unseen/", help="Directory for saving predictions on unseen data")
    parser.add_argument("--input_format", type=str, default="jpg", help="File format of input images")
    parser.add_argument("--output_format", type=str, default="txt", help="File format of output labels")
    parser.add_argument("--template_width", type=int, default=10, help="Template width")
    parser.add_argument("--template_height", type=int, default=15, help="Template height")
    parser.add_argument("--test_image", type=str, default="./sampleCaptchas/input/input21.jpg", help="Path to an input image for testing")
    parser.add_argument("--test_output", type=str, default="./sampleCaptchas/unseen/output21.txt", help="Path to save predicted text output")

    args = parser.parse_args()

    captcha = Captcha(
        template_dir=args.template_dir,
        input_folder=args.input_folder,
        input_format=args.input_format,
        output_folder=args.output_folder,
        output_format=args.output_format,
        template_size=(args.template_width, args.template_height)
    )

    captcha(args.test_image, args.test_output)
