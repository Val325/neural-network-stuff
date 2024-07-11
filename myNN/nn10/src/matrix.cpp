#include <iostream>
#include <vector>
#include <math.h> 
#include <limits.h>
#include <cmath>

class Matrix {       
  public:             
    std::vector<std::vector<double>> mat;
    int height;
    int width;
    int matrixSize;
  Matrix(){}
  Matrix(std::vector<std::vector<double>> mat_inside){
    mat = mat_inside; 
    height = mat_inside.size();
    width = mat_inside[0].size();
    matrixSize = height * width;
  }
  void setMatrix(std::vector<std::vector<double>> mat_inside){
    mat = mat_inside; 
    height = mat_inside.size();
    width = mat_inside[0].size();
    matrixSize = height * width;
  }

  Matrix Transpose(){
    int sizeRow = width;
    int sizeColumn = height;
    
    std::vector<std::vector<double>> output(sizeRow, std::vector<double>(sizeColumn, 0));
    for (int i = 0; i < sizeRow; i++) 
        for (int j = 0; j < sizeColumn; j++) 
            output[i][j] = mat[j][i];  
    
    Matrix m;
    m.setMatrix(output);
    return m; 
  }

  std::vector<double> Flatten(){
    int sizeRow = width;
    int sizeColumn = height;
    
    std::vector<double> output;
    for (int i = 0; i < sizeRow; i++){ 
        for (int j = 0; j < sizeColumn; j++){
            double elemflatten = mat[i][j]; 
            output.push_back(elemflatten);   
        }
    }
    return output; 
  }
  Matrix MultiplyEachElem(double bias){
    int sizeRow = width;
    int sizeColumn = height;
    
    std::vector<std::vector<double>> output(sizeRow, std::vector<double>(sizeColumn, 0));
    for (int i = 0; i < sizeRow; i++) 
        for (int j = 0; j < sizeColumn; j++) 
            output[i][j] = mat[i][j] * bias;  
    
    Matrix m;
    m.setMatrix(output);
    return m; 
  }
  Matrix rotateMatrix180()
  {
    int sizeRow = width;
    int sizeColumn = height;
    Matrix firstTransform;
    firstTransform = Transpose(); 
    //Transpose_inner();
    reverseColumns();
    flip();
    //Transpose_inner();
    //Transpose_inner();
    //reverseColumns();
    return mat;
  }
  void addPadding(int padding_size){
    int sizeRow = width;
    int sizeColumn = height;
    int padding = 2*padding_size;

    std::vector<std::vector<double>> output(sizeRow + padding, std::vector<double>(sizeColumn + padding, 0));

    for (int i = 0; i < sizeColumn; i++) {
        for (int j = 0; j < sizeColumn; j++){
             
            //std::swap(mat[j][i], mat[k][i]);
            output[i+padding_size][j+padding_size] = mat[i][j]; 
            //printf("%d ", mat[i][j]);
        }
        //printf("\n");
    }
    setMatrix(output);
  }
  Matrix Padding(int padding_size){
    int sizeRow = width;
    int sizeColumn = height;
    int padding = 2*padding_size;

    std::vector<std::vector<double>> output(sizeRow + padding, std::vector<double>(sizeColumn + padding, 0));
    for (int i = 0; i < sizeColumn; i++) {
        for (int j = 0; j < sizeColumn; j++){
             
            //std::swap(mat[j][i], mat[k][i]);
            output[i+padding_size][j+padding_size] = mat[i][j]; 
            //printf("%d ", mat[i][j]);
        }
        //printf("\n");
    }
    Matrix m;
    m.setMatrix(output);
    return m;
  }
  void reverseColumns(){
    int sizeRow = width;
    int sizeColumn = height;
    
    std::vector<std::vector<double>> output(sizeRow, std::vector<double>(sizeColumn, 0));
    // Simply print from last cell to first cell.
    for (int i = 0; i < sizeColumn; i++) {
        for (int j = 0, k = sizeRow - 1; j < k; j++, k--){
            std::swap(mat[j][i], mat[k][i]);
            //output[i][j] = mat[i][j]; 
            //printf("%d ", mat[i][j]);
        }
        //printf("\n");
    }
  }
  void flip(){
        for (int j = 0; j < mat.size(); j++) {
            std::reverse(mat[j].begin(), mat[j].end()); 

            /*for (int k = mat[0].size() - 1; k >= 0; k--) {
                std::cout << "j: " << j << std::endl;
                std::cout << "k: " << k << std::endl;

                std::swap(mat[j][k], mat[k][j]);
            }*/
        }
  }
  Matrix operator*(Matrix const& mat_inside){ 
    int sizeRowFirst = mat[0].size();
    int sizeColumn = mat.size();
    int sizeRowTwo = mat_inside.mat[0].size();
    int sizeColumnTwo = mat_inside.mat.size();
    std::cout << "sizeColumn: " << sizeColumn << std::endl;
    std::vector<std::vector<double>> output(sizeRowFirst, std::vector<double>(sizeColumnTwo, 0));
    std::vector<std::vector<double>> outputOne; 
    for (int i = 0; i < sizeRowFirst; i++) {
        for (int j = 0; j < sizeColumn; j++) {
            for (int k = 0; k < sizeRowTwo; k++) {
                if(sizeColumn != 1){
                    output[i][j] += mat[i][k] * mat_inside.mat[k][j];
                }else{
                    output[i][j] += mat[0][k] * mat_inside.mat[k][j];

                }
            }
            if (sizeColumn == 1){
                output[j][i] = output[i][j];
                if (i != 0) output[i][j] = 0;
            }
        }

    }
    Matrix m;
    if (sizeColumn == 1){
        outputOne.push_back(output[0]);
        m.setMatrix(outputOne);
    }else{
        m.setMatrix(output);
    }
    return m;  
  } 
};
  std::ostream& operator<<(std::ostream& out, const Matrix& mat_inside)
  {
    std::cout << "matrix: "  << std::endl;
    if (mat_inside.mat.size() != 1){
        for(int i = 0; i < mat_inside.mat[0].size();i++){
            for(int j = 0; j < mat_inside.mat.size();j++){
                out << mat_inside.mat[i][j] << " ";
            }
            out << "\n";
        } 
    }else{
        for(int j = 0; j < mat_inside.mat[0].size();j++){
            out << mat_inside.mat[0][j] << " ";
        }
    }
        return out;  
  }
