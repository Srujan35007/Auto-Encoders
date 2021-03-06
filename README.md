# Auto-Encoders
Deep generative models especially Auto Encoders and VAEs in both TensorFlow and PyTorch.<br>

## Auto Encoders Linear<br>
### **Status:**<br>
**Pytorch model** - Done.<br>
**Tensorflow model** - In progress. <br>
- MNIST dataset.
- Flattened images.
- Refer [auto_encoder_linear_torch.ipynb](./auto_encoder_linear_torch.ipynb) for Pytorch code.
- Refer [auto_encoder_linear_TF.py](./auto_encoder_linear_TF.py) for Tensorflow code.

#### Model architecture
![flat_encoder_decoder](./images/linear.png)

### Insights
- Linear layers in an encoder decoder setup.
- Good for non image data.
- The number of latent space dimensions drastically changes the accuracy of the model.

### Results
- Flattened images of MNIST dataset were used.
- A Semi-supervised learning method was demonstrated in the code mentioned above.
- Accuracy could not get beyond 80 percent when 5 - 10 percent of the test data was labeled.
- Convolutional neural networks may work better than these linear networks.<br>
___
## Convolutional Auto Encoders<br>
### **Status:**<br>
**Pytorch Model** - Done.<br>
**Tensorflow Model** - In progress.<br>
- Refer [auto_encoders_cnn_torch.py](./auto_encoders_cnn_torch.py) for Pytorch code.
- Refer [auto_encoders_cnn_TF.py](./auto_encoders_cnn_TF.py) for Tensorflow code.

#### Model architecture
![cnn_auto_encoder](./images/cnn.png)
