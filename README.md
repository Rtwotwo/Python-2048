# Python 2048

Python 2048 is based on Alpha Beta pruning MiniMax. This program mainly designs the player mode and AI mode of Python 2048. The AI mode mainly uses the Alpha-Beta pruning algorithm to implement the 2048 game, bringing users a different gaming experience.

## 1.Function Introduction

___Basic Game Rules___: The player moves the number squares with the arrow keys to merge the squares of the same number, earning a score after each merge; The game wins when the number '2048' appears; After each move, a random "2" or "4" is generated in the blank area; If the squares are filled, the game ends.

___Game Mode___: Normal Mode: Players control the game by themselves.
Prompt Mode: Displays recommended movement directions to help players play better; Artificial intelligence mode: The game is automated, using the Minimax algorithm of Alpha-Beta pruning to achieve intelligent decision-making.

___Enhancements___: Aesthetic Enhancement: Added background colors for different numbers and optimized the integral display interface; Sound Control: Added game background music, and the background music is set to random loop playback, and the switch control of game sound effects is added; Dynamic Effects: Added moving dynamic effects to the game's sliders.  
![2048 gui](asserts/游戏界面.png)

## 2.Environments

Operating System: Windows 10/11, MacOS 14/15, Linux series  
Power builder: PyCharm  

  | Index | Name      | Version   | Description                                  |
  |------|-------------|----------|----------------------------------------------|
  | 1    | os          | 3.12.2   | Operating system interface for file and directory manipulation  |
  | 2    | numpy       | 1.26.4   | High-performance numerical computing library that supports array operations |
  | 3    | copy        | 3.12.2   | Deep copy and shallow copy objects |
  | 4    | random      | 3.12.2   | Generate pseudo-random numbers  |
  | 5    | time        | 3.12.2   | Time-related functions such as delay and timing  |
  | 6    | pygame      | 2.5.2    | Game development library with support for graphics and sound  |
  | 7    | sys         | 3.12.2   | Accessing variables and functions of the Python interpreter  |
  | 8    | math        | 3.12.2   | Mathematical functions, such as trigonometric functions and logarithms |

## 3.Function Implementation

![技术实现](asserts/技术实现1.png)
![技术实现](asserts/技术实现2.png)
![技术实现](asserts/技术实现3.png)

## 4.Alpha-Beta pruning principle

The principle of MiniMax algorithm based on Alpha-Beta pruning:  
A.The step implementation of the Minimax algorithm is to determine the maximum search depth D first, which may reach the end or an intermediate pattern;  
B. On the pattern leaf sub-node with maximum depth D, use the predefined value evaluation function to evaluate the leaf node Value is evaluated;  
C. Assign values to non-leaf nodes from the bottom up, where the max node takes the maximum value of the sub-node, and the min node takes the minimum value of the sub-node;  
D. Every time it is our turn (at this time, it must be at a certain max node of the pattern tree), choose the sub-node path whose value is equal to the value of this max node.  
![原理](asserts/原理1.png)  

From this, we borrow the same idea. At present, we need to define an evaluation function to score the number distribution of the squares on the current chessboard. Then, after determining the maximum search depth, move the chessboard in four directions, and then use the comprehensive evaluation function to score the sub-node chessboard obtained in the four directions. Temporarily record the score data in the four directions, and then move the sub-node chessboard after moving.  
![原理](asserts/原理2.png)

During the search process, the algorithm maintains two boundary values: alpha and beta, where alpha represents the lower bound of the best known choice for the current search path, and beta represents the upper bound of the best counterattack strategy that the opponent may adopt. Whenever a node is searched, if the beta value of the current node is found to be less than or equal to the alpha value of its parent node, this means that the search path starting from the current node will not be better than the known best path no matter how it is selected, so the search of this path can be terminated in advance, that is, pruning. In this way, Alpha-Beta pruning effectively reduces the search space and improves the search efficiency, allowing the algorithm to explore deeper search trees in the same time, and then find better solutions.
