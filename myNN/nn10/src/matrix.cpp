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
