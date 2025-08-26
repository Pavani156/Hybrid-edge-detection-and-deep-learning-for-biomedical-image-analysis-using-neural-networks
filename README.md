The project Hybrid Edge Detection and Deep Learning for Biomedical Image Analysis is designed to address the challenges of accurate edge detection in medical images such as MRI, CT, and ultrasound scans. Edge detection plays a critical role in biomedical image processing because it highlights anatomical boundaries, supports segmentation, and assists in medical diagnosis. However, conventional edge detection methods like Sobel, Prewitt, Roberts, and Canny face limitations such as sensitivity to noise, broken or discontinuous edges, and the need for manual parameter tuning.

To overcome these issues, this project proposes a hybrid framework that integrates the strengths of both traditional edge detectors and deep learning models. The approach uses Sobel and Canny operators to extract gradient-based and multi-scale edge information, while a U-Net convolutional neural network (CNN) is trained to refine and enhance edge continuity. The outputs from Sobel, Canny, and U-Net are then fused using a logical OR operation to generate a hybrid edge map, which combines the best features from each method.

The complete workflow involves:

Preprocessing: Converting images to grayscale, applying Gaussian smoothing for noise reduction, and normalization.

Ground Truth Generation: Creating reference edges for evaluation.

Traditional Edge Detection: Applying Sobel and Canny detectors to obtain baseline results.

U-Net Model Training: Training a CNN-based U-Net architecture to learn structural patterns in biomedical images.

Hybrid Edge Fusion: Combining traditional and deep learning results into a single refined edge map.

Testing and Evaluation: Measuring accuracy and robustness of the proposed method.

To validate the approach, multiple performance metrics are used including Accuracy, Precision, Recall, F1-Score, Peak Signal-to-Noise Ratio (PSNR), Mean Square Error (MSE), and Structural Similarity Index Measure (SSIM). The results show that the hybrid method significantly outperforms standalone Sobel or Canny detectors, providing clearer, more continuous, and noise-resistant edges.

This project demonstrates the potential of combining classical image processing with deep learning for biomedical image analysis. It not only improves diagnostic accuracy but also creates a strong foundation for future research in areas such as medical image segmentation, automated disease detection, and computer-aided diagnosis systems.
