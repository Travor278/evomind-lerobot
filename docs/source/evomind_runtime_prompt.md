# 生成 checkpoint 运行清单的 Prompt

复制下面的 Prompt，只替换尖括号变量。目标是生成可执行、可验证的最小清单，不是收集所有可能的信息。

```text
请为目标机器 <HOST> 上的 LeRobot checkpoint <CHECKPOINT_PATH> 创建并验证
<CHECKPOINT_PATH>/evomind-runtime.json。

必须遵守：

1. 读取 config.json、train_config.json、policy_preprocessor.json、policy_postprocessor.json、统计量文件、tokenizer 和模型权重。不要从目录名猜版本。
2. 找到训练环境或已经通过真机验证的推理环境。记录 Python、CUDA、cuDNN、PyTorch 或 JAX/jaxlib、Transformers、Triton、训练精度、AMP、attention backend 和 compile 设置。
3. 特别检查 Transformers 等上层库。对候选环境使用同一 checkpoint、输入、seed 和显式 noise 比较 action；不能只比较延迟。
4. 在 checkpoint 下创建相对入口 runtime/bin/python，并让它指向选定虚拟环境。JSON 只引用 runtime/bin/python，不写机器绝对路径。若使用容器，只允许不可变 image@sha256 digest。
5. 环境必须包含 Evomind worker、机器人插件和相机依赖，例如 pydantic、accelerate、safetensors、piper_sdk、pyrealsense2。不要改变已经验证的数值库版本来补控制依赖。
6. tokenizer 优先使用 <CHECKPOINT_PATH>/tokenizer。训练机遗留的 /root/... 等不可访问路径必须视为 stale path，不能直接使用。
7. 根据 docs/source/evomind_runtime_reference.json 生成最小 JSON。没有影响环境选择的 benchmark 就保留空数组；有正式测量时只保留所选环境的一条记录。
8. 依次验证：manifest schema、网站 inventory、独立 worker preload、worker 实际解释器和版本、机器人插件注册、make_robot_from_config、每路相机单帧读取。相机检查不得连接、使能或移动机械臂。
9. 任一步失败都不得上报“完成”。最终只报告 manifest 路径、选中环境、实际 worker 是否一致和未解决问题，不罗列普通 benchmark 数字。
```

