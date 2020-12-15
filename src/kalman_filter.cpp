#include "kalman_filter.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_*x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  CommonUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  double rho = sqrt(px*px + py*py);
  double rho_dot = 0;
  if (fabs(rho) < 0.00001) {
    px += .001;
    py += .001;
    rho = sqrt(px*px + py*py);
  } 
  rho_dot = (px*vx + py*vy)/rho;
  double theta = atan2(py , px);
  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;

  VectorXd y = z - h;
  for (; y(1) < -M_PI; y(1) += 2*M_PI) {}
  for (; y(1) >  M_PI; y(1) -= 2*M_PI) {}
  CommonUpdate(y);

}

void KalmanFilter::CommonUpdate(const VectorXd &y) {
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();
  x_ = x_ + (K * y);
  I_ = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I_ - K * H_) * P_;
}