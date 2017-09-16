# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program
---
## Reflection

1. Describe the effect each of the P, I, D components had in your implementation.

..* P: This component show oscillation around the CTE and in many case the steering angle is overshooting.

[Video with only P component](video/p.mp4)

..* D This is a very important component helping on the correction of P and reduce the car wobble

[Video with with P+D component](video/pd.mp4)

..* I: This competent is related with the system bias. For this project there is not any steering bias but it may help to smooth the turn when the car is on sharp corners

[Video with with P+I+D component](video/pid.mp4)

2. Describe how the final hyperparameters were chosen.

  Parameters were tuning manually. I first started setting only Kp, Kd and Ki are zero, the value was adjusted until the car was showing oscillation. Then, Kd was tuned until reduce the car overshoot. Finally, there was a small adjustment on Ki to improve the turns of the car on the corners.
