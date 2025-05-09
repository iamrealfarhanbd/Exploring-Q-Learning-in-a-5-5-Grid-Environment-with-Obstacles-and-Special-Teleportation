# Q-Learning Application: Project Setup and Execution Guide

This README file provides comprehensive instructions on how to set up the environment and run the Python script for the Q-learning agent designed to solve the 5x5 grid world navigation problem.

## A. Source Code

[**Note:** In this folder you will find the grid_q_learning.py.]

## B. Execution Instructions

To run the provided Python code and reproduce the results and visualizations presented in the report, please follow these steps:

1.  **Save the Code:**
    Copy all the code from Section A above and paste it into a plain text editor or an Integrated Development Environment (IDE) such as PyCharm. Save the file with a `.py` extension (e.g., `grid_q_learning.py`).

2.  **Required Libraries:**
    The code relies on the following Python libraries. If you do not have these libraries installed, you can install them using `pip`, Python's package management tool.

    * `numpy`: For numerical operations, especially array manipulation for the Q-table.
    * `random`: For generating random numbers for epsilon-greedy exploration.
    * `matplotlib`: For creating the plots and visualizations.
    * `collections`: For using `deque` to track recent rewards for early stopping.
    * `seaborn`: (Used internally by matplotlib for heatmap aesthetics; often installed alongside matplotlib).

    Access your terminal or command prompt and execute these commands:

    ```bash
    pip install numpy matplotlib seaborn
    ```

3.  **Run the Script:**

    * **Using a Terminal:**
        Navigate to the directory where you saved the `grid_q_learning.py` file using your terminal or command prompt. Then, run the script using the Python interpreter:

        ```bash
        python grid_q_learning.py
        ```

    * **Using PyCharm:**
        Open the `grid_q_learning.py` file in PyCharm. You can run the script by clicking the green 'Run' arrow in the top right corner of the IDE or by right-clicking in the editor and selecting 'Run 'grid_q_learning''.

4.  **Observe Output and Plots:**

    * The script will print messages to the console indicating the progress of the experiments for each learning rate.
    * As the script runs, it will generate and display several plot windows sequentially. You will need to close each plot window for the script to proceed to the next one.