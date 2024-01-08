# satDP
Dynamic Programming Algorithm for the Offline Nanosatellite Task Scheduling Problem (ONTS)

## Overview
This repository contains a C++ implementation of a dynamic programming algorithm for solving the Offline Nanosatellite Task Scheduling Problem (ONTS). The algorithm is designed to optimize task scheduling for nanosatellites by considering various constraints and parameters specific to satellite operations.

## Features
- Dynamic programming approach for efficient optimization.
- Parallel computing support to leverage multi-core processors.
- Customizable parameters for versatile application scenarios.

## Requirements
- C++ Compiler (e.g., GCC, Clang)
- OpenMP for parallel processing

## Usage
1. Define the required parameters for the scheduling problem, such as the number of jobs, covers, and time limits.
2. Call the `paralelismo` or `single` functions depending on your need for solving all jobs or a specific one.
3. The program will output the optimized schedule based on the provided parameters.

## Functions
### `pricing_dp`
Calculates the optimal scheduling using dynamic programming.

**Parameters:**
- Job index, total number of jobs, covers, and other scheduling constraints.
- Arrays for shadow prices, allow values, priorities, and various time constraints.

### `paralelismo`
Handles parallel computation for multiple jobs.

**Parameters:**
- Similar to `pricing_dp`, but designed to handle multiple jobs concurrently.

### `single`
Calculates the profile for a single job.

**Parameters:**
- Similar to `pricing_dp`, but tailored for a single job execution.

## License
This project is licensed under GPL v3. Please see the LICENSE file for more details.

## Contributions
Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
