#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if(estimations.size()==0 || estimations.size()!=ground_truth.size()){
      cout << "Error: estimation vector and ground truth vector must be larger than 0 and equal sized" << endl;
      cout << "estimation vector size: " << estimations.size() << endl;
      cout << "ground truth vector size: " << ground_truth.size() << endl;
      return rmse;
  }
  for (unsigned int i=0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array();
    rmse += residual;
  }
  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float px2py2=pow(px,2)+pow(py,2);
  float spx2py2=sqrt(px2py2);
  float px2py2_32=pow(px2py2,3/2);
  // check division by zero
  // taken from lesson solution
  if(fabs(px2py2) < 0.0001){
      cout << "Error: Division by Zero"<<endl;
  }
  
  Hj(0,0)=px/spx2py2;
  Hj(0,1)=py/spx2py2;
  Hj(1,0)=-py/px2py2;
  Hj(1,1)=px/px2py2;
  Hj(2,0)=py*(vx*py - vy*px)/px2py2_32;
  Hj(2,1)=px*(vy*px - vx*py)/px2py2_32;
  Hj(2,2)=px/spx2py2;
  Hj(2,3)=py/spx2py2;
  return Hj;
}
