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
#include "CImg.h"

/*
std::vector<int> OneNum = {1,0,0,0,0,0,0,0,0,0};
std::vector<int> TwoNum = {0,1,0,0,0,0,0,0,0,0};
std::vector<int> ThreeNum = {0,0,1,0,0,0,0,0,0,0};
std::vector<int> FourNum = {0,0,0,1,0,0,0,0,0,0};
std::vector<int> FiveNum = {0,0,0,0,1,0,0,0,0,0};
std::vector<int> SixNum = {0,0,0,0,0,1,0,0,0,0};
std::vector<int> SevenNum = {0,0,0,0,0,0,1,0,0,0};
std::vector<int> EightNum = {1,0,0,0,0,0,0,1,0,0};
std::vector<int> NineNum = {1,0,0,0,0,0,0,0,1,0};
std::vector<int> TenNum = {1,0,0,0,0,0,0,0,0,1};
*/

struct MaxPoolingData
{
    std::vector<std::vector<int>> output;
    std::vector<std::pair<int, int>> MaxPoolBackpropIndex;
    int SizeXMaxPool;
    int SizeYMaxPool;
};

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

std::vector<double> Relu(const std::vector<double>& data) {

    const unsigned long VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        if (data[i] < 0) output[i] = 0;
        if (data[i] >= 0) output[i] = data[i]; 
    }
    
    return output;
}

double Relu(const double& data) {
    double output;
    if (data < 0) output = 0;
    if (data >= 0) output = data; 
    return output;
}

std::vector<double> derivativeRelu(const std::vector<double>& data) {
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);
     
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        if (data[i] < 0) output[i] = 0;
        if (data[i] >= 0) output[i] = 1;
    }
    
    return output;
}

double derivativeRelu(double& data) {
    double output;
    if (data < 0) output = 0;
    if (data >= 0) output = 1; 
    return output;
}

double MSEloss(std::vector<double> X, std::vector<double> Y){
    int sizeOutput = X.size();
    //std::cout << "\nsizeOutput: " << sizeOutput << std::endl;
    //std::cout << "Y: " << Y << std::endl;

    double sum = 0;
    for (int i = 0; i < sizeOutput; i++) {
        //std::cout << " ((double)Y - X[i]) * ((double)Y - X[i]): " <<  ((double)Y - X[i]) * ((double)Y - X[i]) << std::endl;
        sum += (Y[i] - X[i]) * (Y[i] - X[i]);
        //std::cout << "X[i]: " <<  X[i] << std::endl;

        //std::cout << "sum: " << sum << " sizeOutput: " << sizeOutput << " (double)X[i] " << (double)X[i] << " Y: " << Y << std::endl;
    }
    sum = sum / 2*sizeOutput;
    //std::cout << "sum after (sum / sizeOutput): " << sum << std::endl;
    return sum;
}
std::vector<double> MSElossDerivative(std::vector<double> X, std::vector<double> Y){
    int sizeOutput = X.size();
    std::vector<double> output;
    for (int i = 0; i < sizeOutput; i++) {
       double deriv = ((Y[i] - (double)X[i])) / sizeOutput;
       //std::cout << "deriv: " << deriv << " sizeOutput: " << sizeOutput << " (double)X[i] " << (double)X[i] << " Y: " << Y << std::endl;       
       output.push_back(deriv);
    }
    return output;

}

std::vector<double> softmax(const std::vector<double>& data) { 
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);

    double sum = 0; 
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        sum += data[i]; 
    }


    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = data[i] / sum;
    }
    
    return output;
}

//https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
std::vector<std::vector<double>> softmaxDerivative(std::vector<double>& data) { 
    const int VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);
    std::vector<double> softmaxData(VECTOR_SIZE);
    
    softmaxData = softmax(data); 
    std::vector<std::vector<double>> softmaxJacobian(VECTOR_SIZE, std::vector<double>(VECTOR_SIZE, 0)); 
    softmaxJacobian.resize(VECTOR_SIZE);

    for(int i = 0; i != VECTOR_SIZE-1; ++i ) {
       for(int j = 0; j != VECTOR_SIZE-1; ++j ) {
            if (i == j){
                softmaxJacobian[i][j] = softmaxData[i] * (1 - softmaxData[i]);
            }else{
                softmaxJacobian[i][j] = -softmaxData[i] * softmaxData[j]; 
            }
        } 
    }
    
    return softmaxJacobian;
}

std::vector<double> sigmoid(const std::vector<double>& data) {
    
    /*  
        Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        /*if (data[i] < 0){
            //output[i] = 1 / (1 + exp(-data[i]));
            output[i] = exp(data[i]) / (1 + exp(data[i])); 
        }else{
            //output[i] = 1 / (1 + exp(data[i]));
            output[i] = 1 / (1 + exp(-data[i]));
        }*/
         output[i] = 1 / (1 + exp(-data[i]));
        
        /*if (data[i] >= 5.) output[i] = 1.;
        else if (data[i] <= -5.) output[i] = 0.1;
        else output[i] = 1. / (1. + std::exp(-data[i]));
        */
        //output[i] = (1 / (1 + pow(EULER_NUMBER, -data[i])));
    }
    
    return output;
}

double sigmoid(const double& data) {
    double output;
    
    output = 1 / (1 + exp(-data));
    /*
    if (data >= 0){
        output = 1 / (1 + exp(-data));
    }else{
        output = 1 / (1 + exp(data));
    }*/ 
    //output = (1 / (1 + pow(EULER_NUMBER, -data)));
    /*if (data >= 5.) output = 1.;
    else if (data <= -5.) output = 0.1;
    else output = 1. / (1. + std::exp(-data));*/
    //std::cout << "----------------------" << std::endl;
    //std::cout << "1 / (1 + exp(-data)): " << std::endl;
    //std::cout << "sigmoid output: " << output << std::endl;
    //std::cout << "sigmoid data: " << data << std::endl;     
    return output;
}





std::vector<double> derivativeSigmoid(const std::vector<double>& data) {
    /*  
        Returns the value of the derivative sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: derivative sigmoid 
    */ 
    const unsigned long VECTOR_SIZE = data.size();
    std::vector<double> output(VECTOR_SIZE);
    
    
    for(unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[i] = sigmoid(data[i]) * (1 - sigmoid(data[i])); 
    }
    
    return output;
}

double derivativeSigmoid(double& data) {
    /*  
        Returns the value of the derivative sigmoid function f(x) = 1/(1 + e^-x).
        Input: m1, a vector.
        Output: derivative sigmoid 
    */ 
    double output; 
    output = sigmoid(data) * (1 - sigmoid(data));
    //std::cout << "sigmoid(data) * (1 - sigmoid(data)): " << std::endl;
    //std::cout << "sigmoid deriv: " << output << std::endl;
    //std::cout << "sigmoid data: " << data << std::endl; 

    return output;
}



/*
std::vector<std::vector<double>> loadImage(std::string filepath){
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(filepath.c_str(), &width, &height, &bpp, 1);
    const int sizeX = 28;
    const int sizeY = 28;
    std::vector<std::vector<double>> output(sizeX, std::vector<double>(sizeY, 0));  
    int totalPixels = 0;
    for(int i = 0; i < sizeX;i++){
        for(int j = 0; j < sizeY;j++){
            //std::cout << static_cast<unsigned int>(rgb_image[i + totalPixels*j]) << " ";
            output[i][j] = static_cast<double>(rgb_image[i + totalPixels*j]); 
            totalPixels++;
        }
        std::cout << "\n";
    }
    std::cout << "totalPixels: " << totalPixels << std::endl;
    double min = FindMin(output);
    double max = FindMax(output);
    return NormalizeImage(output, 1.0, min, max); 
}*/
/*
std::vector<std::vector<double>> loadImage(std::string filepath){
        //filename = filepath;
        auto inp = OIIO::ImageInput::open(filepath);

        const OIIO::ImageSpec &spec = inp->spec();
        int xres = spec.width;
        int yres = spec.height;

        int nchannels = spec.nchannels;

        auto pixels = std::unique_ptr<int[]>(new int[xres * yres * nchannels]);
        inp->read_image(0, 0, 0, nchannels, OIIO::TypeDesc::UINT8, &pixels[0]);
        inp->close();

        //std::vector<std::vector<std::vector<int>>> Image;

        std::vector<int> GrayArray(xres*yres);
        //std::vector<int> Garray(xres*yres);
        //std::vector<int> Barray(xres*yres);
        
        for (int i=0; i<xres*yres; i++) { 
            GrayArray[i] = pixels[i*nchannels];
            //Garray[i] = pixels[i*nchannels + 1];
            //Barray[i] = pixels[i*nchannels + 2];
        }

        std::vector<std::vector<double>> Grayprocessed(xres, std::vector<double>(yres, 0));
        //std::vector<std::vector<int>> Gprocessed(xres, std::vector<int>(yres, 0));
        //std::vector<std::vector<int>> Bprocessed(xres, std::vector<int>(yres, 0));
        
        int pixelFlatToXY = 0; 
        for(unsigned int x = 0; x != xres; ++x ) {
            for(unsigned int y = 0; y != yres; ++y ) {
                Grayprocessed[x][y] = (double)GrayArray[pixelFlatToXY];
                //Gprocessed[x][y] = (int)Garray[pixelFlatToXY];
                //Bprocessed[x][y] = (int)Barray[pixelFlatToXY];
                pixelFlatToXY++;
            } 
        }

        //Image.push_back(Rprocessed);
        //Image.push_back(Gprocessed);
        //Image.push_back(Bprocessed);
        //return Grayprocessed;
        double min = FindMin(Grayprocessed);
        double max = FindMax(Grayprocessed);
        return Grayprocessed; //NormalizeImage(, 1.0, min, max); 
    }
*/
std::vector<std::vector<double>> loadImage(std::string filepath){
    cimg_library::CImg<unsigned char> img(filepath.c_str());
    int w=img.width();
    int h=img.height();
    int c=img.spectrum();
    //std::cout << "Dimensions: " << w << "x" << h << " " << c << " channels" << std::endl;
    std::vector<std::vector<double>> Image;

    for(int y=0;y<h;y++){
       std::vector<double> xImg;
       for(int x=0;x<w;x++){
           xImg.push_back((double)img(x,y));
           //std::cout << y << "," << x << " " << (double)img(x,y) << std::endl;
       }
       Image.push_back(xImg);
       xImg.clear();
    }
    double min = FindMin(Image);
    double max = FindMax(Image);
    return NormalizeImage(Image, 1.0, min, max);
}
std::vector<std::vector<double>> numLabels = {
    {1,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,1}
};



class ConvolutionalNeuralNetwork {       
    private:
        std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> dataset;
        double learningRate;

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

        std::vector<std::vector<double>> weightsW1;        
        int inputWeights;
        int outputWeights;
        std::vector<double> bias1;        
        int biasSize;

        std::vector<double> denselayerSave;
        std::vector<double> sumLayer;
        
    public:
        ConvolutionalNeuralNetwork(){
            learningRate = 0.001;
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

            inputWeights = 192;
            outputWeights = 10;
            biasSize = 10;
            weightsW1.resize(inputWeights);
            for (int i = 0; i < inputWeights; i++){
                weightsW1[i].resize(outputWeights);
                for (int j = 0; j < outputWeights; j++){
                    weightsW1[i][j] = dist(mt); 
                }
            }
            bias1.resize(biasSize);
            for (int i = 0; i < biasSize; i++){
                bias1[i] = dist(mt); 
            }
        }
    void loadDataset(){
        std::string path = "../dataset/minst/train/";
        /*
        std::string pathZero = "../dataset/minst/train/0";
        std::string pathOne = "../dataset/minst/train/1";
        std::string pathTwo = "../dataset/minst/train/2";
        std::string pathThree = "../dataset/minst/train/3";
        std::string pathFour = "../dataset/minst/train/4";
        std::string pathFive = "../dataset/minst/train/5";
        std::string pathSix = "../dataset/minst/train/6";
        std::string pathOne = "../dataset/minst/train/7";
        std::string pathTwo = "../dataset/minst/train/8";
        std::string pathTwo = "../dataset/minst/train/9";
        std::vector<std::string> pathsNum = {
            "../dataset/minst/train/0", 
            "../dataset/minst/train/1",
            "../dataset/minst/train/2",
        }*/
        int size = 100;
        int iterSize = 0;
        for (int i = 0; i < numLabels.size(); i++){
            std::string pathLoad = path + std::to_string(i);  
            for (const auto & entry : std::filesystem::directory_iterator(pathLoad)){
                if (iterSize >= size) break; 
                
                //std::pair<std::vector<std::vector<int>>, std::vector<double>> data;
                //data = std::make_pair(loadImage(entry.path().string()), numLabels[i]);
                //std::cout << "str: " << entry.path().string() << std::endl;
                std::vector<std::vector<double>> image = loadImage(entry.path().string());
                dataset.push_back(std::make_pair(image, numLabels[i]));
                std::cout << "amount dataset load: " << dataset.size() << std::endl;
                iterSize++;

            }
            iterSize = 0;
        }
        std::cout << "amount dataset: " << dataset.size() << std::endl;
    }
      std::vector<std::vector<double>> convolve(std::vector<std::vector<double>> image, int padding, int stride, std::vector<std::vector<double>> kernelConv, int bias) {


            int sizeW = image.size(); 
            int sizeH = image[0].size();

            int convW = ((sizeW - kernelConv[0].size() + 2 * padding) / stride) + 1;
            int convH = ((sizeH - kernelConv.size() + 2 * padding) / stride) + 1;
            std::vector<std::vector<double>> output(convW, std::vector<double>(convH, 0));
            //std::cout << "convW: " << convW << std::endl;
            //std::cout << "convH: " << convH << std::endl;
        
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
        //std::cout << "sizePoolX: " << sizePoolX << std::endl;
        //std::cout << "sizePoolY: " << sizePoolY << std::endl;
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
      std::vector<double> feedforward(std::vector<std::vector<double>> image){
            //Matrix img;
            //img.setMatrix(image);
            //std::cout << "img: " << std::endl;            
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
                    //std::cout << "number pool: " << numConv+1 << std::endl;
                    std::vector<std::vector<double>> conv2 = convolve(poolLayer1[j], 0, 1, kernelsC2[i], Bias2[i]);
                    convLayer2.push_back(conv2);
                    std::vector<std::vector<double>> pool2 = MaxPool(conv2);
                    Matrix pooltemp;
                    pooltemp.setMatrix(pool2);
                    //std::cout << pooltemp << std::endl;
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
            denselayerSave = DesnseLayer; 
            //std::cout << DesnseLayer.size() << std::endl;
            
            std::vector<std::vector<double>> denselayer;
            denselayer.push_back(DesnseLayer);
            double minDense = FindMin(denselayer);
            double maxDense = FindMax(denselayer);
            denselayer = NormalizeImage(denselayer, 1.0, minDense, maxDense);
            //std::cout << "weightsW1.size(): " << std::endl;
            //std::cout << weightsW1.size() << std::endl;
            //std::cout << "weightsW1[0].size(): " << std::endl;
            //std::cout << weightsW1[0].size() << std::endl;
            std::vector<double> output;
            for (int i = 0; i <  weightsW1[0].size(); ++i){
                double sumNeuron = 0;
                for (int j = 0; j < weightsW1.size(); ++j){
                    sumNeuron = weightsW1[j][i] * DesnseLayer[j]; 
                }
                sumNeuron += bias1[i];
                output.push_back(sumNeuron);
                //std::cout << "output: " << output[i] << std::endl;
            }
            sumLayer = output;
            output = sigmoid(output);
            /*for (int i = 0; i < output.size(); ++i){
                std::cout << i+1 << " : " << output[i] << std::endl;
            }*/
            convLayer1.clear();
            poolLayer1.clear();
            convLayer2.clear();
            poolLayer2.clear();
            DesnseLayer.clear();
            std::vector<double>::iterator result;
            result = std::max_element(output.begin(), output.end());
            return output; 
            //std::cout << "max prob index " << std::distance(output.begin(), result)+1 << std::endl;
            //Matrix dense;
            //dense.setMatrix(denselayer);
           
            //Matrix w1;
            //w1.setMatrix(weightsW1);
            //std::cout << "w1: " << std::endl;
            //std::cout << w1 << std::endl;
            //std::cout << "dense: " << std::endl;
            //std::cout << dense << std::endl;

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
      void train(){
        int epoch = 10;
        //std::cout << "size: " << dataset.size() << std::endl;
        for (int i = 0; i < epoch; ++i){
            for (int j = 0; j < dataset.size(); ++j){
                std::vector<double> prediction = feedforward(dataset[j].first); 
                std::cout << "epoch: " << i << " loss: " << MSEloss(prediction, dataset[j].second) << std::endl;
                
                std::vector<double> derivLoss = MSElossDerivative(prediction, dataset[j].second);
                std::vector<double> sigmoidDeriv = derivativeSigmoid(sumLayer); 
                //std::cout << "derivLoss: " << derivLoss.size() << std::endl;
                //std::cout << "ReluDeriv: " << ReluDeriv.size() << std::endl;
                for (int k = 0; k < derivLoss.size(); ++k){
                    for (int g = 0; g < denselayerSave.size(); ++g){
                        weightsW1[k][g] -= learningRate * derivLoss[k] * sigmoidDeriv[k] * denselayerSave[g];
                    }
                    bias1[k] -= learningRate * derivLoss[k] * sigmoidDeriv[k]; 
                }
                std::vector<std::vector<double>> pools2backprop;
                
                int sizePool = 16;
                for (int i = 0; i < inputWeights / sizePool; ++i){
                    std::vector<double> onePool;
                    for (int j = 0; j < sizePool; ++j){ 
                            onePool.push_back(denselayerSave[i*sizePool+j]); 
                    }
                    pools2backprop.push_back(onePool);
                    onePool.clear();
                }
                //std::cout << "pools2backprop: " << pools2backprop.size() << std::endl;
                //std::cout << "pools2backprop[0]: " << poolLayer2[0].size() << std::endl;
                //std::cout << "pools2backprop[0][0]: " << poolLayer2[0][0].size() << std::endl;

                //std::cout << "softmaxDeriv[0]: " << softmaxDeriv[0].size() << std::endl;

                //denselayerSave
                //std::cout << "softmaxDeriv[0]: " << softmaxDeriv[0].size() << std::endl;
            }
             
        }
      }
};
