#include <iostream>
#include <cnn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "functions.cpp"
#include <bits/stdc++.h>
#include <random>
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

template <class T>
T FindMin(std::vector<std::vector<T>> array){
    T minNumber = INT_MAX;
    for (int i = 0; i < array.size(); ++i){
        for (int j = 0; j < array[0].size(); ++j){
                if (array[i][j] < minNumber) {
                    minNumber = array[i][j];
                }
            }
        }
    
    return minNumber;
}
template <class T>
T FindMax(std::vector<std::vector<T>> array){
    T maxNumber = INT_MIN;
    for (int i = 0; i < array.size(); ++i)              // rows
    {   
        for (int j = 0; j < array[0].size(); ++j){
            if (array[i][j] > maxNumber) {
                maxNumber = array[i][j];
            }
        }
    }
    return maxNumber;
}

template <class T>
std::vector<std::vector<T>> NormalizeImage(std::vector<std::vector<T>> image, T span, T min, T max){
    int sizeX = image.size(); 
    int sizeY = image[0].size();
    std::vector<std::vector<T>> output(sizeX, std::vector<T>(sizeY, 0)); 
    for(unsigned int i = 0; i != sizeX; ++i ) {
       for(unsigned int j = 0; j != sizeY; ++j ) {
            output[i][j] = (span * (image[i][j] - min) / (max-min));
            //std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}

double FindMaxElem(std::vector<std::vector<double>> poolMax){
    double maxNumber = INT_MIN;
    for (int i = 0; i < poolMax[0].size(); ++i)              // rows
    {
        for (int j = 0; j < poolMax.size(); ++j)          // columns
        {
            if (poolMax[i][j] > maxNumber) {
                maxNumber = poolMax[i][j];
            }
        }
    }
    return maxNumber;
}

std::vector<std::vector<double>> loadImage(std::string filepath){
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(filepath.c_str(), &width, &height, &bpp, 3);
    const int sizeX = 28;
    const int sizeY = 28;
    std::vector<std::vector<double>> output(sizeX, std::vector<double>(sizeY, 0));  
    int totalPixels = 0;
    for(int i = 0; i < sizeX;i++){
        for(int j = 0; j < sizeY;j++){
            //std::cout << static_cast<unsigned int>(rgb_image[i + totalPixels*j]) << " ";
            output[i][j] = static_cast<unsigned int>(rgb_image[i + totalPixels*j]); 
            totalPixels++;
        }
        std::cout << "\n";
    }
    std::cout << "totalPixels: " << totalPixels << std::endl;
    double min = FindMin(output);
    double max = FindMax(output);
    return NormalizeImage(output, 1.0, min, max); 
}

class ConvolutionalNeuralNetwork {       
    private:
        std::vector<std::pair<std::vector<std::vector<int>>, int>> dataset;
        
        int numKenrelsC1; 
        int sizeKernelXc1;
        int sizeKernelYc1;
        std::vector<std::vector<std::vector<double>>> kernelsC1;
        std::vector<std::vector<std::vector<double>>> convLayer1;
        std::vector<std::vector<std::vector<double>>> poolLayer1;
        std::vector<double> Bias1;
        
        int numKenrelsC2;         
        int sizeKernelXc2;
        int sizeKernelYc2;  
        std::vector<std::vector<std::vector<double>>> kernelsC2;
        std::vector<std::vector<std::vector<double>>> convLayer2;
        std::vector<std::vector<std::vector<double>>> poolLayer2;        
        std::vector<double> Bias2;
        std::vector<double> DesnseLayer;

    public:
        ConvolutionalNeuralNetwork(){
            std::random_device rd;
            std::mt19937 mt(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0); 
            sizeKernelXc1 = 5;
            sizeKernelYc1 = 5;
            
            sizeKernelXc2 = 5;
            sizeKernelYc2 = 5;

            numKenrelsC1 = 6;
            Bias1.resize(numKenrelsC1);
            for (int i = 0; i < numKenrelsC1; i++){
                Bias1[i] = dist(mt);
            }
            kernelsC1.resize(numKenrelsC1);
            
            std::cout << "---------------" << std::endl; 
            std::cout << "-  kernel1    -" << std::endl; 
            std::cout << "---------------" << std::endl; 
            for (int i = 0; i < numKenrelsC1; i++){
                kernelsC1[i].resize(sizeKernelXc1);
                std::cout << "num kernel: " << i << std::endl;
                for(int j = 0; j < sizeKernelXc1; j++){
                    kernelsC1[i][j].resize(sizeKernelYc1);
                    for(int k = 0; k < sizeKernelYc1; k++){
                        kernelsC1[i][j][k] = dist(mt);
                        std::cout << kernelsC1[i][j][k] << " "; 
                    }
                    std::cout << "\n";
                }
                std::cout << "---------------" << std::endl;
            }

            numKenrelsC2 = 2;
            Bias2.resize(numKenrelsC2);
            for (int i = 0; i < numKenrelsC2; i++){
                Bias2[i] = dist(mt);
            }

            kernelsC2.resize(numKenrelsC2);
            std::cout << "-  kernel2    -" << std::endl; 
            std::cout << "---------------" << std::endl; 
            for (int i = 0; i < numKenrelsC2; i++){
                kernelsC2[i].resize(sizeKernelXc1);
                std::cout << "num kernel: " << i << std::endl;
                for(int j = 0; j < sizeKernelXc2; j++){
                    kernelsC2[i][j].resize(sizeKernelYc1);
                    for(int k = 0; k < sizeKernelYc2; k++){
                        kernelsC2[i][j][k] = dist(mt);
                        std::cout << kernelsC2[i][j][k] << " "; 
                    }
                    std::cout << "\n";
                }
                std::cout << "---------------" << std::endl;
            }


        }
      std::vector<std::vector<double>> convolve(std::vector<std::vector<double>> image, int padding, int stride, std::vector<std::vector<double>> kernelConv, int bias) {


            int sizeW = image.size(); 
            int sizeH = image[0].size();

            int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
            int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;
            std::vector<std::vector<double>> output(convW, std::vector<double>(convH, 0));
            std::cout << "convW: " << convW << std::endl;
            std::cout << "convH: " << convH << std::endl;
        
            int kCenterX = kernelConv.size() / 2;
            int kCenterY = kernelConv.size() / 2;

            for (int h = 0; h < convH; h++){
                for(int w = 0; w < convW; w++){
                    int sum = 0;
                    for (int hk = 0; hk < kernelConv.size(); hk++){
                        int mm = kernelConv.size() - 1 - hk;      
                        for(int wk = 0; wk < kernelConv[0].size(); wk++){
                            int nn = kernelConv.size() - 1 - wk;  

                            int hh = h + (kCenterY - mm);
                            int ww = w + (kCenterX - nn);
                            if (ww-1 >= 0 && ww < convW && hh-1 >= 0 && hh < convH){
                                output[h][w] += image[hh][ww] * kernelConv[mm][nn];
                            }
                        }
                    }
                    output[h][w] += bias;
                }
            }
            return output; 
        }
    std::vector<std::vector<double>> MaxPool(std::vector<std::vector<double>> image){
        int padding = 0;
        int stride = 2;
        int filter = 2;
        
        //MaxPoolingData dataLayer;
        int sizeX = image.size(); 
        int sizeY = image[0].size();
        int sizePoolX = ((sizeX - filter + 2 * padding) / stride) + 1;
        int sizePoolY = ((sizeY - filter + 2 * padding) / stride) + 1;
        std::vector<std::vector<double>> output(sizePoolX, std::vector<double>(sizePoolY, 0));
        //convBeforePooling = convBeforePooling(sizeInputX, std::vector<int>(sizeInputY, 0));
        std::cout << "sizePoolX: " << sizePoolX << std::endl;
        std::cout << "sizePoolY: " << sizePoolY << std::endl;
        for (int i = 0; i < sizePoolX; ++i)              // rows
        {
            for (int j = 0; j < sizePoolY; ++j)          // columns
            {
                std::vector<std::vector<double>> pool(filter, std::vector<double>(filter, 0));
                if (i-1 >= 0 && i < sizeX && j-1 >= 0 && j < sizeY){
                    pool[0][0] = image[i*stride][j*stride]; 
                    pool[0][1] = image[i*stride][j*stride+1];
                    pool[1][0] = image[i*stride+1][j*stride];
                    pool[1][1] = image[i*stride+1][j*stride+1];

                    //std::vector<int> indexs = FindMaxIndex(pool);
                    //std::pair<int, int> pair(indexs[0], indexs[1]);
                    //std::pair<std::pair<int, int>, int> pairOut(pair, FindMaxElem(pool)); 
                    //PoolingData.MaxPoolBackpropIndex.push_back(pairOut);
                    //std::cout << "--------------------------" << std::endl;
                    //std::cout << "i: " << indexs[0] << std::endl;
                    //std::cout << "j: " << indexs[1] << std::endl;

                    //std::cout << "pool[0][0]: " << pool[0][0] << std::endl;
                    //std::cout << "pool[0][1]: " << pool[0][1] << std::endl; 
                    //std::cout << "pool[1][0]: " << pool[1][0] << std::endl;
                    //std::cout << "pool[1][1]: " << pool[1][1] << std::endl;
                    //std::cout << "max: " << FindMaxElem(pool) << std::endl;
                    output[i][j] = FindMaxElem(pool);
                } 
            }
        }

        return output;
    }
      void feedforward(std::vector<std::vector<double>> image){
            //Matrix img;
            //img.setMatrix(image);
            //std::cout << img << std::endl;
            for (int i = 0; i < numKenrelsC1; ++i){
                std::vector<std::vector<double>> conv1 = convolve(image, 0, 1, kernelsC1[i], Bias1[i]);
                convLayer1.push_back(conv1);
                std::vector<std::vector<double>> pool1 = MaxPool(conv1);
                poolLayer1.push_back(pool1);
            }
            
            int numConv = 0;
            for (int i = 0; i < numKenrelsC2; ++i){
                for (int j = 0; j < poolLayer1.size(); ++j){
                    std::cout << "number pool: " << numConv+1 << std::endl;
                    std::vector<std::vector<double>> conv2 = convolve(poolLayer1[j], 0, 1, kernelsC2[i], Bias2[i]);
                    convLayer2.push_back(conv2);
                    std::vector<std::vector<double>> pool2 = MaxPool(conv2);
                    Matrix pooltemp;
                    pooltemp.setMatrix(pool2);
                    std::cout << pooltemp << std::endl;
                    poolLayer2.push_back(pool2);
                    numConv += 1;
                }
            }
            for (int i = 0; i < poolLayer2.size(); ++i){
                for (int j = 0; j < poolLayer2[0].size(); ++j){
                    for (int k = 0; k < poolLayer2[0][0].size(); ++k){
                        DesnseLayer.push_back(poolLayer2[i][j][k]); 
                    }
                }
            }            
            std::cout << DesnseLayer.size() << std::endl; 
            //Matrix pool;
            //pool.setMatrix(pool2);
            //std::cout << pool << std::endl;
            /*
            std::vector<std::vector<std::vector<double>>> conv1layer;
            for (int i = 0; i < numKenrelsC1; i++){
                std::vector<std::vector<double>> convOne = convolve(image, 0, 1,kernelsC1[i]);
                conv1layer.push_back(convOne);
            }
            for (int i = 0; i < numKenrelsC1; i++){
                std::cout << "---------------" << std::endl; 
                std::cout << "- kernel1 conv-" << std::endl; 
                std::cout << "---------------" << std::endl;
                for (int h = 0; h < conv1layer[i].size(); h++){
                    for(int w = 0; w < conv1layer[i][h].size(); w++){
                        std::cout << conv1layer[i][h][w] << " ";
                    }
                    std::cout << "\n";
                }
        }*/
      }
};
