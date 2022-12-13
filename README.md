# CEMPendulum
使用CEM规划算法实现Pendulum的运动控制

## 使用
pendulum_model.py 用于定义信念运动模型

CEM.py 用于实现CEM规划算法，规划配置是config/planner.yaml，用yaml读取入PlannerCfg类

main.py 调用gym的pendulum，实际的执行


## 说明
状态转移模型目前使用的是绝对精确（直接从gym中复制出来状态转移函数），未来计划使用Koopman，基于数据学习模型

奖励模型目前也是绝对精确的，这个可能尝试Koopman，或是直接神经网络。