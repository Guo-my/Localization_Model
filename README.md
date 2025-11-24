### A Three-component Localization Method for A Single Station Based on Deep Neural Networks

> [!NOTE]
> The code is implemented in PyTorch, requiring pre-configuration of the corresponding environment.

### This figure displays a prediction result from the model on the test set, with the right-side heatmap illustrating the back-azimuth predicted by the classification model.
![A localization example](https://github.com/Guo-my/Localization_Model/blob/main/Figure/Localization_example.png)

# Article abstract
This study proposes a deep neural network-based approach for single-station three-component seismic localization. The method employs two networks that takes three-component waveforms as input to predict epicentral distance and back-azimuth respectively. Completely data-driven, our approach requires no prior conditions or manual intervention. We trained the model using a dataset containing 367k high-quality seismic events, followed by systematic evaluation across three dimensions: (1) performance comparison with mainstream localization models, (2) robustness testing under low SNR conditions, and (3) generalization assessment using K-Net data. Experimental results demonstrate that our method achieves high-precision single-station three-component localization in most scenarios, validating its effectiveness and practical utility. Furthermore, we conducted an in-depth comparison between regression and classification models for back-azimuth prediction, finding that the classification model maintains marginally higher errors than the regression model, but it can provide reliability estimates for predicted back-azimuth. This research provides a novel technical solution for advancing seismic monitoring capabilities, particularly in network-sparse regions and for small events monitoring task.
