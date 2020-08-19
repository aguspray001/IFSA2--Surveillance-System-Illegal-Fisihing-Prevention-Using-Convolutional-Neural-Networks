# IFSA2 (Illegal Fishing Smart Amphibious Aircraft): Surveillance System Illegal Fisihing Prevention Using Convolutional Neural Networks

Indonesia has a vast ocean with an abundance of fishes with its natural environments. Those abundances have to be conserved to prevent further destruction of the environment, which can result in the extinction of the surrounding living things. The government had deployed a vessel monitoring system, but illegal fishing still hardly been controlled. In this paper, toward conserving the fishes and especially the environment, we present a surveillance system framework from aerial images using drones technology. We develop a surveillance system using only visual information from the camera installed on the UAV and the design of the convolutional layer for accurate detection. Parameters are learned automatically because the learning process is pure from visual data that learned, so that makes the surveillance and investigation process easier. Experiment show relatively well that the proposed method successfully reaches Average Precision (AP)=75.03%, and hull plate classification reaches Average Matching Precision (AMP)=96.44%, and we believe it could bring many benefits for the ministry of fisheries and marine affairs Indonesia for identifying the illegal vessels and reduce the number of illegal fishing.

<p align="center">
  <img width="800" height="400" src="https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/Demo.gif">
</p>

## Overall System of IF-SA2 (Illegal Fishing Smart Amphibious Aircraft): Surveillance System Illegal Fisihing Prevention Using Convolutional Neural Networks

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/full%20process.png)

for clearly explanation, i divide the overall system to be spesific block diagram system.

## UAV Proses

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/UAV%20PROSES.png)

UAV task is bring the camera for online streaming or mapping mode, and send to the server (firebase) via online.

## Server Proses

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/SERVER%20PROSES.png)

The server tasks:
1. in the first step, UAV will be scanning on the conservative area to capture the vessels on the ocean.
2. and then, UAV will send the data (captured images) to the serve, and server will process the data.
3. the results from this system are mapping area (orthoimage), ship location (based on GPS), and Hull plate number classification.

## Overall System Has Been Conducted by Simulation Before Real Application

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/Result/simulation/Screenshot%20from%202020-06-05%2014-52-14.png)

Simulation system has been conducted by using V-REP Coppelia platform, in this simulation i've add the object detection on the 3D simulation.

## Graphical User Interface of IF-SA2

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/Result/GUI/gambungan.png)

**(a) Graphical User Interface for Mapping and Detection System and (b) Graphical User Interface for Hull Plate Number Classification**

**output from maping and detection system is an interactive map that is shown below:**

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/map.png)

1. The red mark  : Detected vessels.
2. The green mark: AIS data from satellite (dummy, *because AIS data is for commercial uses*).

**The hull number plate classification has been conducted by using the feature match method (SIFT (Scale-Invariant Feature Transform) algorithm for feature extraction and KNN (K-Nearest Neighbor) for match the feature), this is the visualization for clearly explanation:**

![alt text](https://github.com/aguspray001/IFSA2--Surveillance-System-Illegal-Fisihing-Prevention-Using-Convolutional-Neural-Networks/blob/master/Result/feature%20match/gabung.png)

*If you are interested on my project you can add this on your citation and download my dataset in this link: [link to my dataset!](https://intip.in/IFSA2dataset). This project focused on the computer vision, deep learning, and machine learning method.*
