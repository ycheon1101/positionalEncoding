# positionalEncoding

When reconstruct the image with pure mlp, the reconstructed image looks like blur image. It is because losing the contextual or location information by passing the multi layer perceptron. To come up with this problem, we have to do positional encoding. By doing positional encoding before passing the model, we can give the spatial information to the model. 

So, I used Gaussian Fourier Transform(sinusoid) to do positional encoding.

