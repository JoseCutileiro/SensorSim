# SensorSim

Sensor simulations to test the quality of different trajectory prediction algorithms

# Basic Tests

In this section, we evaluate different algorithms under three distinct scenarios. The first scenario assumes a perfect sensor as well as a flawless tracking algorithm, with no errors or failures. The second scenario introduces slight errors in the positional readings of objects but does not simulate sensor failures. Finally, the third scenario accounts for both positional errors and sensor measurement failures.

# Conclusions from the Basic Tests:

Assuming a perfect sensor, Extended Kalman Filters generally perform better. This is because they achieve significantly superior results compared to other algorithms when dealing with nonlinear trajectories. However, their more conservative nature can be a disadvantage in scenarios where the trajectories are strictly linear. In addition to this significant advantage of EKF, this type of predictor is capable of handling noise. Although the noise slightly impacts the results, the EKF still performs remarkably well. It is important to note that the simulations involve a noise level of 15%, which is relatively high. Nevertheless, it is not a perfect algorithm, and when a high rate of failures is introduced (33% of measurements are discarded), it is unable to provide a timely response.

# What Should We Do, Then?

There are three possible approaches we can take:

1. Refine the input data, aiming to mitigate the impact of noise and handle failures more effectively.

2. Improve the prediction algorithm, potentially by combining multiple algorithms or identifying one that has been overlooked.
   
3. Investigate the noise and failure thresholds in a real-world system to determine whether the limitations of EKF are even relevant for the development of SafeRoadside.

*Important Note:* If we follow the intuition of option 2, it is highly relevant to develop a system that learns trajectories over time, leveraging the fact that the RSU is in a fixed position. However, this approach has a drawback: it may disregard rare movements, such as reverse driving. Nonetheless, it would operate in a more general manner, and the system would become "smarter" over time.*