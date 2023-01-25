# Super Resolution

This project of super resolution was built by Xianyao Chen, when serving for the GPU chip's company (siliconarts, 주식회사 실리콘아츠) in South Korea.

[1] Target: Upscale 2k video to 4k video 

[2] Background: Given 2k (1920 x 1080) video, upscale 2k to 4k (2160, 3840) video with deep learning technique.

[3] Techniques:

3.1. Overparameterization: Collapse a very large deep network into a highly efficient deep network. 

     3.1.1 The model is trained on RGB channels, and upscale 1920 x 1080 image to 2160 x 3840 image in inference stage
     
     3.1.2 pixel shuffle: Rearrange elements of image

3.2. In Super_Resolution/model_structure/, training_model.png and test_model.png are the training model and the test model, respectively.

[4] Result: Output 4k images and display 4k images

4.1 In Super_Resolution/output/proposal_model/, the proposal model output 4k images.

4.2 In Super_Resolution/output/reference/, the reference output are listed.

4.3 In Super_Resolution/, the performance.png shows the performances of proposal model. For example, "p_1" indicates the "proposal_output_1.bmp" under Super_Resolution/output/proposal_model/; "r_1" indicates the "reference_1.bmp" under Super_Resolution/output/reference/
    
    4.3.1 The average of SSIM values is 0.8001. Each frame can be inferred within 45.37 millisecond by using NVIDIA GeForce RTX 3070 Ti.
   
[5] In order to test the robustness of the proposal model, 2k video ("sample_1920x1080.mkv") can be downloaded from URL: https://filesamples.com/formats/mkv#google_vignette   

    5.1 Execute the following command to test the robustness of the proposal model: python test_demo.py --vid_path sample_1920x1080.mkv 
    




