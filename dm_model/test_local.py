import sys
import os
import json
import io

# Get the current directory (where test_local.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the 'code' directory to the Python path
code_dir = os.path.join(current_dir, 'code')
sys.path.append(code_dir)  # This line makes sure Python knows about the 'code' directory
sys.path.append(os.path.join(code_dir, 'networks'))

# Now we can import from 'code/inference.py'
from inference import model_fn, input_fn, predict_fn, output_fn
from PIL import Image

def test_local_inference():
    # Set model_dir to the directory containing model.pth
    model_dir = os.path.join(current_dir, 'model')
    model_path = os.path.join(model_dir, 'model.pth')
    print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    model, norm_type, patch_size = model_fn(model_dir)

    # List of test images
    test_images = ['photoshop.jpg', 'real.jpg', 'fake-repub.jpeg', 'pope.png', 'pope1.png', 'pope2.jpg', 'test_image.png']

    for image_name in test_images:
        image_path = os.path.join(current_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Test image not found: {image_path}")
            continue

        print(f"\nProcessing image: {image_name}")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Simulate the SageMaker inference pipeline
        input_data = input_fn(io.BytesIO(image_bytes), 'application/x-image')
        prediction = predict_fn(input_data, (model, norm_type, patch_size))
        result = output_fn(prediction, 'application/json')

        print(f"Result for {image_name}:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_local_inference()
