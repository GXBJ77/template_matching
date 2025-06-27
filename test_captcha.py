import unittest
import os
from glob import glob
from captcha import Captcha

class TestCaptchaFullDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_folder = "./sampleCaptchas/input/"
        cls.output_folder = "./sampleCaptchas/output/"
        cls.input_format = "jpg"
        cls.output_format = "txt"
        cls.template_size = (10, 15)
        cls.template_dir = "./sampleCaptchas/templates/"
        cls.captcha = Captcha(
            template_dir=cls.template_dir,
            template_size=cls.template_size,
            input_folder=cls.input_folder,
            input_format=cls.input_format,
            output_folder=cls.output_folder,
            output_format=cls.output_format
        )

        images = sorted(glob(os.path.join(cls.input_folder, f"*.{cls.input_format}")))
        labels = sorted(glob(os.path.join(cls.output_folder, f"*.{cls.output_format}")))

        # only keep the pairs that both images and labels exist
        def extract_suffix(filename, prefix):
            base = os.path.splitext(os.path.basename(filename))[0]
            return base.replace(prefix, "")

        image_dict = {extract_suffix(f, "input"): f for f in images}
        label_dict = {extract_suffix(f, "output"): f for f in labels}

        common_suffixes = sorted(set(image_dict) & set(label_dict))

        cls.matched_images = [image_dict[suffix] for suffix in common_suffixes]
        cls.matched_labels = [label_dict[suffix] for suffix in common_suffixes]

    def test_all_labelled_predictions(self):
        failed_cases = []

        for input_path, label_path in zip(self.matched_images, self.matched_labels):
            filename = os.path.basename(input_path)

            if not os.path.exists(label_path):
                print(f"⚠️ Skipping {filename}: No label file found.")
                continue

            with open(label_path, 'r') as f:
                expected_label = f.read().strip()

            prediction = self.captcha.match_characters(input_path)

            if prediction != expected_label:
                failed_cases.append((filename, expected_label, prediction))

        if failed_cases:
            failure_message = "\n".join([
                f"{fname}: expected='{expected}', predicted='{pred}'"
                for fname, expected, pred in failed_cases
            ])
            self.fail(f"{len(failed_cases)} CAPTCHA predictions failed:\n{failure_message}")
        else:
            print(f"✅ All {len(self.matched_images)} labeled CAPTCHA predictions passed.")


    def test_unseen_captchas_21_jpg_format(self):
        input_path = "./sampleCaptchas/input/input21.jpg"

        prediction = self.captcha.match_characters(input_path)
        self.assertEqual(prediction, "CL69V")

        print(f"✅ {input_path} unseen CAPTCHA predictions passed. Prediction: {prediction}")

    def test_unseen_captchas_21_txt_format(self):
        input_path = "./sampleCaptchas/input/input21.txt"

        prediction = self.captcha.match_characters(input_path)
        self.assertEqual(prediction, "CL69V")

        print(f"✅ {input_path} unseen CAPTCHA predictions passed. Prediction: {prediction}")

    def test_unseen_captchas_100(self):
        input_path = "./sampleCaptchas/input/input100.jpg"

        prediction = self.captcha.match_characters(input_path)
        self.assertEqual(prediction, "YMB1Q")

        print(f"✅ {input_path} unseen CAPTCHA predictions passed. Prediction: {prediction}")

if __name__ == "__main__":
    unittest.main()
