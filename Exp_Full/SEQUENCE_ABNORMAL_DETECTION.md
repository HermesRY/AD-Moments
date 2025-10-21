# Exp2 异常片段检测 - 基于原始Sequence时间戳

## 功能说明

✅ **已实现**: 直接基于原始 `sequence` 数据检测异常片段，并输出对应的时间戳

## 核心特点

1. **无需预提取特征**: 直接处理原始 `sequence` 数据 (action_id + timestamp)
2. **使用CTMS模型**: 利用4个encoder (circadian, task, movement, social) 进行异常检测
3. **输出原始时间戳**: 异常片段直接对应原始sequence中的events和timestamps

## 运行方式

```bash
cd /home/heming/Desktop/AD-Moments-1/AD-Moments/Exp_Full/Without_Personalization
python3 exp2_abnormal_detection.py
```

## 输出文件

### 1. `exp2_abnormal_sequences_CI.json`
CI组受试者的异常片段检测结果

### 2. `exp2_abnormal_sequences_CN.json`  
CN测试组受试者的异常片段检测结果

## 输出格式

```json
{
  "threshold": 2.022,  // 异常阈值(95th percentile)
  "threshold_percentile": 95,
  "total_subjects_analyzed": 36,
  "subjects_with_abnormalities": 11,  // 有异常的受试者数量
  "abnormal_records": [
    {
      "subject_id": 3,
      "label": "CI",
      "total_windows": 83,  // 总窗口数
      "abnormal_windows": 12,  // 异常窗口数
      "abnormal_percentage": 14.5,  // 异常百分比
      "mean_score": 1.179,  // 平均异常分数
      "max_score": 2.168,  // 最大异常分数
      "time_range": {  // 受试者数据的时间范围
        "min_timestamp": 1673961436,
        "max_timestamp": 1675212794
      },
      "abnormal_segments": [  // 所有异常片段
        {
          "window_idx": 18,  // 窗口索引
          "event_start_idx": 180,  // 在sequence中的起始位置
          "event_end_idx": 210,  // 在sequence中的结束位置
          "score": 2.026,  // 该窗口的异常分数
          "num_events": 30,  // 该窗口包含的事件数
          "timestamp_range": {  // 该窗口的时间范围
            "min": 1674757947,
            "max": 1674764007
          },
          "sequence_events": [  // 🎯 原始sequence事件列表
            {
              "ts": 1674757947,
              "action_id": 7
            },
            {
              "ts": 1674758147,
              "action_id": 7
            },
            ...
          ]
        }
      ]
    }
  ]
}
```

## 关键字段说明

### 异常判定
- **threshold**: 基于95th百分位数自动计算
- **score > threshold**: 该窗口被标记为异常

### 时间戳信息
每个异常片段包含:
1. **timestamp_range**: 该异常片段的时间范围(min/max Unix timestamp)
2. **sequence_events**: 该片段中所有原始events，包含:
   - `ts`: Unix timestamp
   - `action_id`: 动作ID (1-22)

### 窗口参数
- `SEQ_LEN = 30`: 每个窗口30个事件
- `STRIDE = 10`: 滑动步长10个事件
- 窗口重叠: 20个事件 (66.7%)

## 工作流程

### 1. 数据加载
```python
sequence = [
    {"ts": 1673961436, "action_id": 19},
    {"ts": 1673961436, "action_id": 2},
    ...
]
```

### 2. 转换为模型输入
```python
activity_ids = [18, 1, ...]  # action_id - 1 (转为0-indexed)
hours = [14.5, 14.5, ...]  # 从timestamp提取小时数(0-23.99)
```

### 3. 创建滑动窗口
```python
windows_ids = [
    [18, 1, 12, ...],  # window 0: events 0-29
    [5, 7, 19, ...],   # window 1: events 10-39
    ...
]
```

### 4. CTMS编码 & 异常检测
```python
for each window:
    circ_emb = model.circadian_encoder(window_ids, window_hours)
    task_emb = model.task_encoder(window_ids)
    move_emb = model.movement_encoder(window_ids)
    soc_emb = model.social_encoder(window_ids)
    
    # 计算z-score
    z_circ = |norm(circ_emb) - baseline_mean| / baseline_std
    ...
    
    # 加权异常分数
    score = 0.5*z_circ + 0.3*z_task + 0.1*z_move + 0.1*z_soc
```

### 5. 提取异常片段的时间戳
```python
if score > threshold:
    # 找到对应的原始sequence events
    abnormal_events = sequence[window_start:window_end]
    # 提取timestamps
    timestamps = [e['ts'] for e in abnormal_events]
```

## 实际应用示例

### 读取异常结果
```python
import json
from datetime import datetime

with open('exp2_abnormal_sequences_CI.json', 'r') as f:
    data = json.load(f)

# 查看第一个有异常的受试者
subject = data['abnormal_records'][0]
print(f"Subject {subject['subject_id']}: {subject['abnormal_windows']} abnormal windows")

# 查看第一个异常片段
seg = subject['abnormal_segments'][0]
print(f"\nAbnormal segment #{seg['window_idx']}:")
print(f"Score: {seg['score']:.3f}")
print(f"Events: {seg['num_events']}")

# 转换时间戳为可读格式
min_time = datetime.fromtimestamp(seg['timestamp_range']['min'])
max_time = datetime.fromtimestamp(seg['timestamp_range']['max'])
print(f"Time range: {min_time} to {max_time}")

# 查看该片段的所有events
for event in seg['sequence_events']:
    ts = datetime.fromtimestamp(event['ts'])
    print(f"  {ts}: Action {event['action_id']}")
```

### 分析异常模式
```python
# 统计CI组异常情况
ci_abnormal_counts = []
for subject in data['abnormal_records']:
    if subject['label'] == 'CI':
        ci_abnormal_counts.append(subject['abnormal_percentage'])

print(f"Average abnormal percentage in CI: {np.mean(ci_abnormal_counts):.1f}%")

# 找出异常分数最高的片段
all_segments = []
for subject in data['abnormal_records']:
    for seg in subject['abnormal_segments']:
        seg['subject_id'] = subject['subject_id']
        all_segments.append(seg)

all_segments.sort(key=lambda x: x['score'], reverse=True)
top_segment = all_segments[0]
print(f"\nMost abnormal segment:")
print(f"Subject: {top_segment['subject_id']}, Score: {top_segment['score']:.3f}")
```

## 模型参数

- **d_model**: 64 (embedding dimension)
- **Alpha weights**: [0.5, 0.3, 0.1, 0.1]
  - Circadian: 50%
  - Task: 30%
  - Movement: 10%
  - Social: 10%
- **Device**: CUDA (if available)
- **Baseline**: 从CN training set (14个受试者) 计算

## 运行结果示例

```
================================================================================
EXP2: ABNORMAL SEGMENT DETECTION WITH SEQUENCE TIMESTAMPS
================================================================================

1. Loading and processing dataset...
   CN train: 14, CN test: 4, CI: 36

2. Initializing CTMS model...
   Device: cuda
   Alpha weights: [0.5, 0.3, 0.1, 0.1]

3. Computing baseline from CN train...
   Baseline statistics:
     circadian: mean=3.9984, std=0.5438
     task: mean=0.4779, std=0.0009
     movement: mean=0.7238, std=0.0492
     social: mean=8.5311, std=6.0975

4. Detecting abnormal segments with sequence timestamps...
   Abnormal threshold (95th percentile): 2.0221

RESULTS:
CI subjects with abnormalities: 11/36 (30.6%)
CN subjects with abnormalities: 1/4 (25.0%)
```

## 技术优势

✅ **直接使用原始数据**: 不需要预处理特征提取
✅ **保留完整时间信息**: 异常片段包含精确的Unix timestamps
✅ **可解释性强**: 每个异常片段都有对应的action序列
✅ **高效处理**: 使用GPU加速，批量处理
✅ **统计严谨**: 基于正常人群(CN)建立baseline

## 下一步分析建议

1. **时间模式分析**: 分析异常片段是否集中在特定时间段(早晨/夜晚)
2. **行为模式分析**: 统计异常片段中最常见的action_id组合
3. **个体差异**: 比较不同CI受试者的异常模式
4. **临床关联**: 将异常百分比与临床评分(MoCA, ZBI等)关联分析
5. **可视化**: 绘制异常片段在时间轴上的分布图
