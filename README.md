# TEMPLATE MATCHING
***********
# TM_CCOEFF : Template matching Cross-Correlation Coefficient

TM_CCOEFF is a method used in template matching, a technique used in computer vision and image processing to find a template image within a larger image.

Template matching works by sliding a template image (a smaller image) over a larger image and calculating a similarity metric at each position. TM_CCOEFF (Cross-Correlation Coefficient) is one of the methods used to calculate this similarity metric.  In TM_CCOEFF, the similarity between the template and the image patch under the template is measured by computing the correlation coefficient. It considers the mean-subtracted pixel values in the template and the corresponding region of the image. The correlation coefficient ranges from **-1** to **1**, where **1** indicates a **perfect match**, **-1** indicates a **perfect negative match**, and **0 indicates no correlation**.


                        T = [[1, 2],
                             [3, 4]]

                        I = [[0, 1, 2, 3],
                             [4, 5, 6, 7],
                             [8, 9, 1, 2],
                             [3, 4, 5, 6]]


                        I_sub at poistion 0,0 
                               = [[0, 1],
                                 [4, 5]]

                        μ(T) = (1 + 2 + 3 + 4) / 4 = 2.5
                        μ(I_sub) = (0 + 1 + 4 + 5) / 4 = 2.5

                        CCC = Σ[(T[i, j] - μ(T)) * (I_sub[i, j] - μ(I_sub))] / sqrt[Σ(T[i, j] - μ(T))^2 * Σ(I_sub[i, j] - μ(I_sub))^2]

                        CC = ((1-2.5)*(0-2.5) + (2-2.5)*(1-2.5) + (3-2.5)*(4-2.5) + (4-2.5)*(5-2.5)) / sqrt[((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) * ((0-2.5)^2 + (1- 2.5)^2 + (4-2.5)^2 + (5-2.5)^2)]


***********
# TM_SQDIFF

                        T = [[1, 2],
                             [3, 4]]

                        I = [[0, 1, 2, 3],
                             [4, 5, 6, 7],
                             [8, 9, 1, 2],
                             [3, 4, 5, 6]]

                        I_sub at poistion 0,0 
                               = [[0, 1],
                                 [4, 5]]

                        Squared differences = [(1-0)^2, (2-1)^2, (3-4)^2, (4-5)^2] = [1, 1, 1, 1]

                        Sum of squared differences = 1 + 1 + 1 + 1 = 4


This value represents the degree of dissimilarity between the template and the image patch at that position.

This process is repeated for other positions as well. The position with the lowest sum of squared differences indicates the best match between the template and the image.
