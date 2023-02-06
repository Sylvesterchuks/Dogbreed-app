
### 1. Imports and class names setup ###
import gradio as gr
import os
import numpy as np
import torch
import torchvision.transforms as T

from model import Efficient_b2_model
from timeit import default_timer as timer
from typing import Tuple, Dict
from data_setup import classes, model_tsfm

# Setup class names
#class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation ###
#test_tsfm = T.Compose([T.Resize((224,224)),
#                        T.ToTensor(),
#                       T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
#                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
#                       ])

# Create EffNetB2 Model
effnet_b2 = Efficient_b2_model(num_classes=len(classes), pretrained=True)
#effnet_b2
#effnetb2, test_transform = create_effnet_b2(num_of_class=len(class_names), 
                            #transform=test_tsfm,
                            #seed=42)

# saved_path = 'demos\foodvision_mini\09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth'
saved_path = 'efficient_b2_checkpoint_model_2023_02_04.pth'

print('Loading Model State Dictionary')
# Load saved weights
effnet_b2.load_state_dict(
                torch.load(f=saved_path,
                           map_location=torch.device('cpu'), # load to CPU
                          )
                        )

print('Model Loaded ...')
### 3. Predict function ###

# Create predict function
from typing import Tuple, Dict

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    #img = get_image(img_path, model_tsfm).unsqueeze(0)
    img = model_tsfm(image=np.array(img))["image"]
    img = img.unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnet_b2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnet_b2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {classes[i]: float(pred_probs[0][i]) for i in range(len(classes))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio App ###

# Create title, description and article strings
title= 'DogBreed Mini 🐩🐶🦮🐕‍🦺'
description = "An EfficientNetB2 feature extractor computer vision model to classify images of Dog breeds."
article = "<p>ImageWoof Created by Chukwuka </p><p style='text-align: center'><a href='https://github.com/Sylvesterchuks/foodvision-app'>Github Repo</a></p>"


# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type='pil'), # What are the inputs?
                    outputs=[gr.Label(num_top_classes=10, label="Predictions"), # what are the outputs?
                             gr.Number(label='Prediction time (s)')], # Our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article
                   )
# Launch the demo
print('Gradio Demo Launched')
demo.launch()

