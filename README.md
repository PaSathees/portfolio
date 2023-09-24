# AI Project Portfolio

Welcome to my AI project portfolio! This repository showcases a collection of machine learning and deep learning projects I've worked on using PyTorch, TensorFlow, scikit-learn, and MLOps tools. Each project includes a brief description, links to the code, and a summary of the key technologies used.

## Projects

| Project | Description | Technologies | Code | Demo | Key Features |
| ------- | ----------- | ------------ | ---- | ---- | ------------ |
| [FoodVision Big](PYTORCH/PyTorch_Model_Deployment.ipynb) | Deployed a 101 food items classification PyTorch Model | HuggingFace, Gradio, PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_Model_Deployment.ipynb) | [Demo](https://huggingface.co/spaces/PaSathees/FoodVision_Big) | - Created a FoodVision Big model for classification of 101 food items by fine-tuning EfficientNet_B2 pre-trained model on Food101 dataset. Deployed FoodVision Big App made with Gradio to Production using HuggingFace Spaces. |
| [FoodVision Mini](PYTORCH/PyTorch_Model_Deployment.ipynb) | Deployed a Pizza, Steak, Sushi classification PyTorch Model | HuggingFace, Gradio, PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_Model_Deployment.ipynb) | [Demo](https://huggingface.co/spaces/PaSathees/FoodVision_Mini) | - Created a FoodVision Mini model for classification of Pizza, Steak, and Sushi by fine-tuning EfficientNet_B2 pre-trained model. Deployed FoodVision Mini App made with Gradio to Production using HuggingFace Spaces. |
| [PyTorch Vision Transformer (ViT) Paper Replication](PYTORCH/PyTorch_ViT_Paper_Replication.ipynb) | Replicated Original Vision Transformer Model with PyTorch | PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_ViT_Paper_Replication.ipynb) | [Demo Notebook](PYTORCH/PyTorch_ViT_Paper_Replication.ipynb) | - Replicated Vision Transformer (ViT) model (ViT-Base), Patch Embedding, Class Token Embedding, Position Embedding, Multi-Head Self Attention (MSA), Multilayer Perceptron (MLP), Custom Transformer Encoder, PyTorch Transformer Layers, Complete ViT model, Training on Custom ViT model, Getting and fine-tuning a Pre-Trained ViT model (ViT_B_16), Plotting loss curves, Predictions and evaluation |
| [PyTorch Experiment Tracking](PYTORCH/PyTorch_Experiment_Tracking.ipynb) | Experiment multiple PyTorch models. | TensorBoard, PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_Experiment_Tracking.ipynb) | [Demo Notebook](PYTORCH/PyTorch_Experiment_Tracking.ipynb) | - Tracked expermentation of multiple PyTorch model trainings with model architectures (EfficientNet_B0, EfficientNet_B2), epochs, dataset size. Visualized experiments results with TensorBoard, Selected best model for predictions |
| [PyTorch Transfer Learning](PYTORCH/PyTorch_Transfer_Learning.ipynb) | Applying transfer learning using PyTorch. | PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_Transfer_Learning.ipynb) | [Demo Notebook](PYTORCH/PyTorch_Transfer_Learning.ipynb) | - Learned how to apply transfer learning on a TorchVision Pre-trained model (efficientnet_b0), made predictions and evaluated model metrics (loss, accuracy, precision, recall, F1 score, Confusion Matrix, Classification Report) |
| [PyTorch Custom Dataset](PYTORCH/PyTorch_Custom_Dataset.ipynb) | Creating custom dataset for PyTorch for images and building a model using it. | PyTorch, TorchVision, Python | [GitHub Location](PYTORCH/PyTorch_Custom_Dataset.ipynb) | [Demo Notebook](PYTORCH/PyTorch_Custom_Dataset.ipynb) | - Learned how to create dataset using ImageFolder and Custom dataset class. Created dataloaders, Transfer learning on a TorchVision Pre-trained model (efficientnet_b0), Analyzed predictions |
| [Neural Network with Numpy](NUMPY/NN-WITH-NUMPY/NN_with_Numpy.ipynb) | Building ANN (Artificial Neural Networks) from scratch and testing it on a standard dataset. | Python, Numpy | [GitHub Folder](NUMPY/NN-WITH-NUMPY) | [Demo Notebook](NUMPY/NN-WITH-NUMPY/NN_with_Numpy.ipynb) | - Learned under the hood operations of ANN models such as Weight initialization, activations, forward propagation, loss functions (binary cross entropy, categorical cross entropy), backward propagation, updating parameters, making predictions, testing with dummy dataset and MNIST data. |

<!-- ## MLOps

In addition to individual projects, I have experience with MLOps (Machine Learning Operations) practices and tools:

- **Continuous Integration/Continuous Deployment (CI/CD)**: Describe your experience with CI/CD pipelines for ML projects.
- **Containerization**: Mention your use of Docker or other containerization technologies.
- **Model Versioning**: Explain how you manage model versions and deployments.
- **Monitoring and Logging**: Share your approach to monitoring and logging in production ML systems.
- **Automated Testing**: Discuss your strategies for testing ML models and pipelines.
- **Deployment**: Explain how you deploy models into production environments. -->

## About Me

My name is Sathees Paskaran, and I am a recent graduate in IT Data Science. I have a strong passion for innovation and applying cutting-edge technologies for the greater good. I have experience working on more than 8 campus projects as a team lead, and I have worked as a Software Engineering (BI, Data Engineering) intern at iTelaSoft Pvt. Ltd.

One of my most successful projects was CarboMeter, which I researched and led myself. This project won me recognition as a finalist at the Imagine Cup 2023. However, I am now focused on building a career in AI and applying it to real-world problems. To achieve this, I have completed several MOOC courses (which are listed on my LinkedIn profile) and am currently learning PyTorch, Tensorflow, Scikit-Learn, and MLOps. I am building this repository to showcase my portfolio projects to recruiters and connections.

## Contact Information

Please feel free to contact me on [LinkedIn](https://www.linkedin.com/in/sathees-paskaran/) or email (sathpaskaran@gmail.com), and have a look at my [GitHub profile](https://github.com/PaSathees) or [my personal website](https://pasathees.github.io/). Looking forward to connect with you.


## References

These are some of the references I have used when learning and implementing my AI portfolio projects here. I hope my journey will help some others too.

Updated on: 11 Sept 2023

| Problem Area           | Datasets                                        | Real-World Applications                           | SOTA (State of the Art)                                               | Learning Resources                                                       |
|------------------------|-------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------|
| Image Classification   | - [CIFAR-10](link_to_cifar10), <br> - [ImageNet](link_to_imagenet) | - [Medical Image Diagnosis](link_to_medical_image_app), <br> - [Object Recognition](link_to_object_recognition) | - [SOTA](link_to_classification_sota) | -  |
|||| - [SOTA](link) | - |
| Natural Language Processing | - [IMDb reviews](link_to_imdb_dataset), <br> - [Twitter sentiment](link_to_twitter_sentiment) | - [Sentiment Analysis](link_to_sentiment_analysis_app), <br> - [Chatbots](link_to_chatbots) | - [SOTA](link_to_nlp_sota) | -  |
| Object Detection  | - [COCO dataset](link_to_coco_dataset) | - [Surveillance Systems](link_to_surveillance_app),<br> - [Autonomous Vehicles](link_to_autonomous_vehicles) | - [SOTA](link_to_object_detection_sota)  | -  |
| Generative Adversarial Networks (GANs) | - [CelebA](link_to_celeba_dataset), <br> - [MNIST](link_to_mnist_dataset) | - [Image Generation](link_to_image_generation_app),<br> - [Deepfake Detection](link_to_deepfake_detection) | - [SOTA](link_to_gan_sota) | - |
| Reinforcement Learning | - [OpenAI Gym environments](link_to_openai_gym) | - [Agent Training in Environments](link_to_agent_training) | - [SOTA](link_to_rl_sota) | -  |
| Time Series Forecasting | - [Stock price data](link_to_stock_data), <br> - [weather data](link_to_weather_data) | - [Stock Price Prediction](link_to_stock_prediction_app), <br> - [Weather Forecasting](link_to_weather_forecasting) | - [SOTA](link_to_ts_forecasting_sota) | -  |
| Anomaly Detection      | - [Credit card fraud data](link_to_fraud_data) | - [Anomaly Detection in Financial Transactions](link_to_fraud_detection), <br> - [Network Intrusion Detection](link_to_network_anomaly_detection) | - [SOTA](link_to_anomaly_detection_sota) | -  |
| Recommendation Systems | - [MovieLens](link_to_movielens), <br> - [Amazon reviews](link_to_amazon_reviews) | - [Movie Recommendations](link_to_movie_recommendations), <br> - [E-commerce Product Recommendations](link_to_ecommerce_recommendations) | - [SOTA](link_to_recommender_sota) | -  |
| Semantic Segmentation  | - [Pascal VOC](link_to_pascal_voc), <br> - [Cityscapes](link_to_cityscapes) | - [Semantic Image Segmentation](link_to_semantic_segmentation), <br> - [Autonomous Driving](link_to_autonomous_driving) | - [SOTA](link_to_segmentation_sota)  | - 
| Speech Recognition     | - [LibriSpeech](link_to_librispeech), <br> - [Common Voice](link_to_common_voice) | - [Speech-to-Text Conversion](link_to_speech_to_text), <br> - [Voice Assistants](link_to_voice_assistants) | - [SOTA](link_to_asr_sota) | -
-  |

<!-- 
| Item       | Description  | Price |
|------------|--------------|-------|
| Product 1  | Description1 | $10   |
|            | Description1.1 | $5  | 
|            | Description1.2 | $5  |  
| Product 2  | Description2 | $20   |
|            | Description2.1 | $10 |  
|            | Description2.2 | $10 |  
| Product 3  | Description3 | $30   |

-->
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for detail

---

