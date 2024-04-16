from scorer import Scorer
import json
class VQAScorer(Scorer):
    def __init__(self, text, images, idx, prompt_path):
        super().__init__(text, images)
        with open(prompt_path,"r") as file:
            self.prompts = json.load(file)
        self.idx = idx
    def _calculate_score(self):
        scores = []
        for _, image_result in self.prompts[self.idx]["images_vqa_scores"].items():
            scores.append(image_result)
        self.scores = scores