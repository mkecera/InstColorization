The most common loss function to measure the similarity between two images is to calculate the average of similarity per pixel (pix2pix). Common pix2pix regression loss function include L1, L2, and smoothed-L1(Huber) loss, defined mathematically in the following equations:
$$
L2 = \frac{1}{n} \sum_n(\hat{x}_i-x_i)^2
$$

$$
L1 = \frac{1}{n} \sum_n |\hat{x}_i-x_i|
$$

$$
\text{Huber Loss} = \frac{1}{n} \sum_{n}\left(\begin{array}{c}
0.5 \times\left(\hat{x}_{i}-x_{i}\right)^{2} \text { if }\left|\hat{x}_{i}-x_{i}\right|<1 \\
\left|\hat{x}_{i}-x_{i}\right|-0.5 \text { otherwise }
\end{array}\right)
$$

Where $\hat{x}_i$ represents the predicted and $x_i$ represents the ground truth pixel value in channel a or b. Loss from channel a and channel b are summed up. We first experimented with these three pix2pix loss functions and find the most suitable one.

The image coloring problem itself is an under-constrained multi-modal problem, and there is no right or wrong answer. The MSE loss function reaches its minimum value at the average of the training set. Therefore, it predicts the mean of multiple possible coloring schemes, result in desaturated predictions. In addtion, we divided a and b color space into $23\times23$ bins and predict the probability distribution per pixel over the bins. We introduced cross-entropy loss per pixel.

We also observed unnatural color patches by observing the output of baseline model, expecially at the boundary of the instance bounding boxes, as shown in figure. We get the intuition that if two neighboring pixels have similar lightness (values in the L channel), they're more likely to have similar colors. Therefore, we also derived "Neighbor Loss" to integrate spatial localily in the loss function. aims for smoother change in color per pixel.
$$
\text{Neighbor Loss} = e^{-(L_p-L_q)^2} [(\hat{a}_p-\hat{a}_q) + (\hat{b}_p-\hat{b}_q) ],  q\in N(p)
$$
我们最终采用的损失函数是上述三种的线性组合。

The MSE loss function reaches its minimum value at the average of the training set. Therefore, it predicts the mean of multiple possible coloring schemes, result in desaturated predictions.
Among the three, huber loss is known for its robustness and low sensitivity to outliers, which makes it suitable for detection in this situation



To evaluate the performance of each loss function, we trained the model with 100 epochs. The first 40 epochs use learning rate of 5$\times{10}^{-5}$. Afterwards we train for 60 more epoch with decayed learning rate (linearly decayed from 5$\times{10}^{-5}$ to 0). Due to the variant number of detected bounding boxes in images, we used batch size = 1. Adam optimization is adopted with $\beta 1=0.9, \beta 2 = 0.999$.



In the first step, we evaluate the performance of three pix2pix regression loss functions, $l1, l2$ and smoothed-$l1$ (Huber) loss.
Figure \ref{fig:pix2pix_loss} shows the loss function change over 100 epochs. Due to the multi-modality and ambiguity of the coloring problem, we are not surprised to find that the L1 and L2 loss functions do not show a clear decrease with training. Huber loss however, solves the averaging problem since it's a robust estimator and converges with more training. 
In addition, we illustrate the PSNR/SSIM metrics change over epochs in figure \ref{fig:pix2pix_metric}. As shown in the learning curve, the training based on the L1 and L2 loss functions has hardly learned anything.  The Huber loss-based metrics curve however, performs differently than the loss function curve. On the training set, both metrics steadily grows as the epoch increases; on the validation set, they reach their peak after 6 epoch. Overfitting appeared very early, which is because instead of random initialization, we trained on the basis of pre-trained weights by Zhang et al.

Based on the image, we selected Huber loss as the pix2pix regression loss, which was trained together with the other two loss functions.



In the next step, we first take the distribution of colors in different intervals into consideration, and then also take into account the local similarity.
Based on the respective scales of the three loss functions and some initial experiments, we finally determined our composite loss G = CrossEtropy loss + 10$\times$Huber loss + + 10$\times$Neighbor loss. Figure x and Figure y show change of loss function two metrics with epoch respectively. As more aspects are considered in the loss function, the degree of overfitting becomes less, indicating that our model obtained more robustness and generalizability.

Finally, we select the best trained model based on performance on PSNR and SSIM metrics on validation set and evaluate their performance on the test set. The two selected models are indicated by red dots in Figure. Model A is trained with Huber loss for the 6 epoch, and moedel B is trained with combination of three loss functions for 36 epoch. The averaged performance are shown in Table. 

Although our two best models are not as good as the baseline model overall due to the limited training size and computing resources, we can still see that the composite loss function performs better in terms of generalization ability. In addition, our model performs better in certain specific instances, as shown in the figure. In the first example, the ice cream was successfully colored. In the second example, the colors at the door handle blend better between instance and background and avoid unnatural color patches. This may be due to  advantage of taking local similarity into consideration. Although the model trained based on Huber loss can also be colored fluently, it still shows overall desaturation.



Contributed aspects: Data preprocessing, model implementation, loss function implementation, experiments, user test, visualization, writeup

Details: 

- Preprocessed train/validation/test dataset 
- Implemented model structure modification to include per-pixel classification output
- Implemented loss function including L1, L2, Huber, per-pixel cross entropy and local similarity loss
- Implemented the validation pipeline for calculating loss / metrics per epoch
- Initiated 3 independent user test
- Trained and evaluated model performance for 6 different loss functions
- Produced Figure 8-12. Writeup the report for approach, experiments and results.