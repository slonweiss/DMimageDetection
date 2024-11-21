import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from normalization import get_list_norm, CenterCropNoPad
from normalization2 import PaddingWarp
from get_method_here import get_method_here
import io
import logging

def create_model(model_name, model_dir):
    # Get model parameters
    _, _, arch, norm_type, patch_size = get_method_here(model_name, model_dir)
    
    # Load the checkpoint from the expected path
    model_path = os.path.join(model_dir, model_name, 'model_epoch_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract the model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Unexpected checkpoint structure")

    # Create the model based on the architecture
    if arch == 'res50stride1':
        from networks import resnet_mod
        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Load the state dict
    model.load_state_dict(state_dict)
    
    return model, norm_type, patch_size

def model_fn(model_dir):
    model, norm_type, patch_size = create_model('Grag2021_progan', model_dir)
    model.eval()
    return model, norm_type, patch_size

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    print(f"Content type received: {request_content_type}")
    print(f"Request body type: {type(request_body)}")
    
    if request_content_type == 'application/json':
        import json
        import base64
        
        try:
            # If request_body is bytes or bytearray, decode to string first
            if isinstance(request_body, (bytes, bytearray)):
                request_body = request_body.decode('utf-8')
            
            # Parse the JSON
            json_obj = json.loads(request_body)
            image_b64 = json_obj.get('image')
            if not image_b64:
                raise ValueError("No 'image' key found in request JSON")
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_b64)
            
            # Convert to BytesIO and open with PIL
            image_data = io.BytesIO(image_bytes)
            image = Image.open(image_data).convert('RGB')
            
            print(f"Successfully loaded image of size: {image.size}")
            return image
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            raise
            
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(image, model_info):
    model, norm_type, patch_size = model_info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = []
    if patch_size is not None:
        if patch_size > 0:
            transform.append(CenterCropNoPad(patch_size))
        else:
            transform.append(CenterCropNoPad(-patch_size))
            transform.append(PaddingWarp(-patch_size))

    transform += get_list_norm(norm_type)
    transform = transforms.Compose(transform)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor).cpu().numpy()

    if output.shape[1] == 1:
        logit = output[0, 0]
    elif output.shape[1] == 2:
        logit = output[0, 1] - output[0, 0]
    else:
        raise ValueError("Unexpected output shape")

    if len(output.shape) > 2:
        logit = np.mean(logit)

    return logit

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        probability = 1 / (1 + np.exp(-prediction))
        return {
            'logit': float(prediction),
            'probability': float(probability),
            'is_fake': bool(probability > 0.5)
        }
    raise ValueError(f'Unsupported content type: {content_type}')
