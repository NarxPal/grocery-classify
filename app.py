import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
model = torch.load('./model.pth')  # Replace with your model path
model.eval()

# Define the image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

def predict(image):
    image = transform(image).unsqueeze(0)  # Preprocess the image
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    predicted_class = class_names[predicted.item()]  # Get class label
    return predicted_class

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Preprocess the image
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    predicted_class = class_names[predicted.item()]  # Get class label
    return predicted_class


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Accepts image input
    outputs="text",  # Output is text (predicted class)
    title="Fruit & Vegetable Classifier",
    description="Upload an image of a fruit or vegetable to get a prediction."
)

# Launch the interface
iface.launch()
