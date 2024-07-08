#include <iostream>
#include <cnn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "functions.cpp"
#include <bits/stdc++.h>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebufalgo.h>

#include "matrix.cpp"
#include "convolution.cpp"

int main(){
    std::vector<std::vector<double>> matrix4x4 = {
        {4, 4, 4, 4},
        {4, 4, 4, 4},
        {4, 4, 4, 4},
        {4, 4, 4, 4}
    };
    std::vector<std::vector<double>> matrix4x1 = {
        {2, 4, 4, 2},
        {2, 4, 4, 2},
        {2, 4, 4, 2},
        {2, 4, 4, 2},
    };
    Matrix matqqw;
    matqqw.setMatrix(matrix4x4);
    Matrix matone;
    matone.setMatrix(matrix4x1);
    std::cout << matone << std::endl;
    
    //std::cout << matone.Transpose() << std::endl;
    ConvolutionalNeuralNetwork cnn;
    Matrix img;
    std::vector<std::vector<double>> image = loadImage("../dataset/minst/test/7/0.jpg");
    img.setMatrix(image);
    std::cout << img << std::endl;
    cnn.feedforward(image);
    /*
    for(int i = 0; i < image[0].size();i++){
        for(int j = 0; j < image.size();j++){
            std::cout << image[i][j] << " ";
        }
        std::cout << "\n";
    }*/ 
}
