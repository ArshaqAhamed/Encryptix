import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import nltk

nltk.download('punkt')

# Load pre-trained ResNet model for feature extraction
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, embed_size):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, x):
        with torch.no_grad():
            features = self.resnet(x)
            features = features.view(features.size(0), -1)
            features = self.fc(features)
        return features

# LSTM model for caption generation
class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        # Concatenate features and embeddings
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

# Function to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

# Function to generate captions (simplified for demonstration)
def generate_caption(feature_extractor, caption_generator, image_path, vocab):
    image = preprocess_image(image_path)
    features = feature_extractor(image)
    captions = torch.LongTensor([[vocab['<start>']]])
    caption_generated = []

    for i in range(20):  # max caption length
        outputs = caption_generator(features, captions)
        _, predicted = outputs.max(2)
        predicted_id = predicted[0, -1].item()
        if predicted_id == vocab['<end>']:
            break
        captions = torch.cat((captions, torch.LongTensor([[predicted_id]])), 1)
        caption_generated.append(vocab_inv[predicted_id])
    
    return ' '.join(caption_generated)

# Example usage
if __name__ == "__main__":
    # Define vocabulary (simplified for demonstration purposes)
    vocab = {'<start>': 0, '<end>': 1, 'a': 2, 'dog': 3, 'on': 4, 'the': 5, 'beach': 6}
    vocab_inv = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    # Initialize models
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    feature_extractor = ResNetFeatureExtractor(embed_size)
    caption_generator = CaptionGenerator(embed_size, hidden_size, vocab_size, num_layers)
    
    # Load pre-trained weights if available (for caption generator)
    # caption_generator.load_state_dict(torch.load('caption_generator.pth'))
    
    # Generate caption for an image
    image_path = "D:\OneDrive - University of Hertfordshire\ML\Encryptix Internship\images.jpeg" # Replace with your image path
    caption = generate_caption(feature_extractor, caption_generator, image_path, vocab)
    print(f"Generated Caption: {caption}")
