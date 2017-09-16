#include "kalman_filter.h"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;  

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  //Calculate the z_pred ourself (Check seesion 14 lesson 5)
  // Check 17 Radar Measurment

  std::cout << "x_ = " << x_ << std::endl;

  float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
  float theta = atan2(x_(1) , x_(0));  //Normalizing Angles use atan2 to returns values between -pi and pi.
  float rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;

  VectorXd z_pred = VectorXd(3);
  z_pred << rho, theta, rho_dot;


  //In C++, atan2() returns values between -pi and pi. When calculating phi 
  //in y = z - h(x) for radar measurements, the resulting angle phi in the y vector should 
  //be adjusted so that it is between -pi and pi

  VectorXd y = z - z_pred;

  
  if (y(1) > 3.1416) {
      y(1) -= 6.2831;
  }

  if (y(1) <= -3.1416) {  //y(1) -> phi from measurment
      y(1) += 6.2831;
  }  
  
  /*  
  std::cout << "theta = " << theta << std::endl;
  std::cout << "z = " << z << std::endl;
  std::cout << "z_pred = " << z_pred << std::endl;
  std::cout << "y(1) = " << y(1) << std::endl;
  */

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;  



}
