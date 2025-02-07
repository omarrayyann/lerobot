import torch
from PIL import Image
import torchvision.transforms as transforms
import requests

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config

OBS_ROBOT = "observation.state"
ACTION = "action"
TASK = "task"
OBS_IMAGE = "observation.image"

def main():

    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    config = PI0Config(input_features=input_features, output_features=output_features)

    # pi0 policy
    policy = PI0Policy.from_pretrained("lerobot/pi0").to("cuda")
    policy.config = config

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    dummy_image = Image.open(requests.get(url, stream=True).raw)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dummy_image = transform(dummy_image).unsqueeze(0).to("cuda")
    dummy_state = torch.rand(1, 10).to("cuda")
    dummy_task = ["Pick the pen from the table"]

    dummy_batch = {
        OBS_IMAGE: dummy_image,
        OBS_ROBOT: dummy_state,
        TASK: dummy_task,
    }

    action, generated_text = policy.select_action(dummy_batch)

    print("action:", action)
    print("generated_text:", generated_text)
    
if __name__ == "__main__":
    main()
