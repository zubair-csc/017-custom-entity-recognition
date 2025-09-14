# 🔍 Custom Entity Recognition - Transformers-based NER Model Training

## 📋 Project Overview
Custom Entity Recognition is a comprehensive and easy-to-use implementation for building Named Entity Recognition (NER) models using Transformers in PyTorch. Built with clean architecture and modular design, it includes complete data preparation, model training, evaluation, and inference pipelines. The system supports training custom NER models from scratch with extensive customization options for entity types, datasets, and model configurations.

## 🎯 Objectives
- Build custom Named Entity Recognition models for domain-specific applications
- Implement transformer-based NER architecture following best practices and stable training techniques
- Support multiple entity types (Organizations, Persons, Locations) with flexible label schemes
- Provide real-time training monitoring with loss visualization and model performance tracking
- Enable model saving/loading functionality for reproducibility and deployment
- Offer configurable hyperparameters for different training scenarios and entity recognition tasks

## 📊 Dataset Information
| Attribute | Details |
|-----------|---------|
| **Supported Format** | IOB (Inside-Outside-Begin) tagging format |
| **Entity Types** | Organizations (ORG), Persons (PER), Locations (LOC) |
| **Input Format** | Tokenized text with corresponding entity labels |
| **Label Scheme** | BIO format (B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC, O) |
| **Processing** | Automatic tokenization, label alignment, and batch processing |
| **Tokenization** | Subword tokenization with proper label alignment for transformer models |

## 🔧 Technical Implementation

### 🏗️ Model Architecture
- **Base Model**: DistilBERT for efficient training and inference
- **Classification Head**: Token classification layer for sequence labeling
- **Tokenization**: WordPiece tokenization with subword alignment
- **Label Handling**: Automatic label-to-ID mapping with ignore tokens for subwords
- **Architecture**: Transformer encoder with classification head for each token

### 🧹 Data Preprocessing
**Preprocessing Pipeline:**
- Automatic tokenization with subword alignment
- Label synchronization with tokenized inputs
- Padding and truncation for batch processing
- Special token handling ([CLS], [SEP], [PAD])
- Ignore index (-100) for subword tokens and special tokens

### ⚙️ Training Architecture
**Training Process:**
1. **Data Preparation**: Convert text and labels to model-compatible format
   - Tokenize input sentences with word-level alignment
   - Map entity labels to numerical IDs
   - Handle subword tokenization with proper label alignment

2. **Model Training**: Fine-tune transformer model for NER task
   - Token-level cross-entropy loss calculation
   - Adam optimizer with learning rate scheduling
   - Gradient clipping and proper weight initialization

3. **Training Features**:
   - Cross-entropy loss for multi-class token classification
   - Automatic mixed precision training support
   - Model checkpointing and best model selection
   - Comprehensive metrics tracking and evaluation

### 📈 Training Features
**Monitoring and Evaluation:**
- Real-time loss tracking during training epochs
- Token-level accuracy calculation with proper masking
- Entity-level precision, recall, and F1-score metrics
- Training progress visualization and statistics display

**Model Management:**
- Automatic model saving after training completion
- Load pre-trained models for inference and continued training
- Generate predictions on new text with confidence scores
- Configurable hyperparameters and training settings

## 📊 Visualizations
- **Training Progress**: Loss curves and accuracy metrics over epochs
- **Entity Predictions**: Highlighted text with entity labels and confidence scores
- **Model Performance**: Confusion matrices and classification reports
- **Prediction Examples**: Interactive entity recognition results display

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: PyTorch, transformers, datasets, numpy

### Installation
1. Clone the repository:
```bash
git clone https://github.com/zubair-csc/017-custom-entity-recognition.git
cd 017-custom-entity-recognition
```

2. Install required libraries:
```bash
pip install torch transformers datasets numpy
```

### Dataset Setup
The system includes sample training data and supports custom datasets:
- **Built-in Data**: Sample entities for organizations, persons, and locations
- **Custom Data**: Easy integration with your own labeled datasets
- **Format**: IOB tagging format with tokenized sentences

### Running the Training
Execute the Python script:
```python
python custom_ner_training.py
```

This will:
- Load sample training data automatically
- Initialize DistilBERT model for token classification
- Train the NER model for specified epochs
- Display training progress and performance metrics
- Save trained model and generate prediction examples

## 📈 Usage Examples

### Basic Training
```python
# Initialize and train NER model
trainer = SimpleNERTrainer()
model, tokenizer = trainer.train()
```

### Generate Predictions
```python
# Make predictions on new text
trainer.test_model()
trainer.quick_test("Apple Inc. is located in California")
```

### Custom Entity Types
```python
# Add your own entity types to training data
CUSTOM_DATA = [
    {
        "tokens": ["Tesla", "Motors", "manufactures", "electric", "vehicles"],
        "labels": ["B-ORG", "I-ORG", "O", "O", "O"]
    }
]
```

### Load Pre-trained Model
```python
# Load and use saved model
tokenizer = AutoTokenizer.from_pretrained("./ner-model-final")
model = AutoModelForTokenClassification.from_pretrained("./ner-model-final")
```

### Custom Configuration
```python
# Modify training parameters
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    num_train_epochs=5,
    learning_rate=3e-5
)
```

## 🔮 Future Enhancements
- Multi-language NER model support for cross-lingual entity recognition
- Nested entity recognition for complex entity hierarchies
- Active learning integration for efficient data annotation
- Real-time entity recognition API with REST endpoints
- Integration with popular annotation tools for dataset creation
- Advanced evaluation metrics and model comparison tools
- Few-shot learning capabilities for new entity types
- Production-ready deployment scripts and containerization

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
- **Transformers Library** by Hugging Face for the transformer implementations
- **PyTorch** for the deep learning framework
- **DistilBERT** for efficient transformer architecture
- Open source community for continuous support and inspiration

## 📞 Contact
Zubair - [GitHub Profile](https://github.com/zubair-csc)

Project Link: [https://github.com/zubair-csc/017-custom-entity-recognition](https://github.com/zubair-csc/017-custom-entity-recognition)

⭐ Star this repository if you found it helpful!
