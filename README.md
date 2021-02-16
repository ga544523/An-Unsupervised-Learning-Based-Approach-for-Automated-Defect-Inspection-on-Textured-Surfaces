# multi-scale-CDAE

referenced from  and the dataset is from DAGM2007



[An Unsupervised-Learning-Based Approach for Automated Defect Inspection on Textured Surfaces](https://ieeexplore.ieee.org/document/8281622)



Add random noise on different scale images and then try to recover it to orginal image in 8x8 patch size(see image1).
Using Multi scale image just for increase the robustness(I guess)

![image](https://github.com/ga544523/multi-scale-CDAE/blob/main/result_figure/Untitled%20Diagram.png?raw=true)


If the recorved patch has the large difference with the orginal one which means this patch has high probolaity being a defect.(see result below with gamma=2.0)


![image](https://github.com/ga544523/multi-scale-CDAE/blob/main/result_figure/result1.png?raw=true)






![image](https://github.com/ga544523/multi-scale-CDAE/blob/main/result_figure/result3.png?raw=true)






![image](https://github.com/ga544523/multi-scale-CDAE/blob/main/result_figure/result4.png?raw=true)

