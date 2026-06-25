# MPDD-AVG 2026 Young 多模态抑郁检测

本仓库用于提交语音信息处理期末大作业相关材料，围绕 MPDD-AVG 2026 Young Track 的多模态抑郁检测任务展开。

## 目录结构

```text
code/
  repro_scripts/                         # 可复现脚本
  official_baseline/make_submission_forcodabench/
                                           # 官方提交格式与最终提交产物
docs/
  final_assignment_latex/                 # 大作业文档 LaTeX 源文件与图片资源
  pdf/最终大作业文档.pdf                   # 最终大作业 PDF
presentation/
  语音抑郁者检测_visual_optimized_cover_tuned_with_agenda_fixed.pptx
                                           # 汇报 PPT
```

## 复现说明

代码入口位于 `code/repro_scripts/run_pipeline.py`。在具备数据目录和依赖环境时，可使用：

```powershell
cd code
$env:BLEND_PHQ_THRESHOLD='4.25'
$env:TERNARY_T2='11.0'
$env:OUT_NAME='young_final_t4p25_t2p11'
python repro_scripts\run_pipeline.py
```

如需从原始数据重新训练四路面部 ranker 和 ORIG 主干：

```powershell
cd code
$env:SKIP_CACHED='1'
python repro_scripts\run_pipeline.py
```

依赖见 `code/repro_scripts/requirements.txt`。

## 数据与权重说明

本仓库不包含 MPDD 原始数据集，不包含模型权重。完整重训需要用户本地提供 `Train-MPDD-Young/Young/` 与 `Test-MPDD-Young/Young/` 数据目录。

## 文档

最终大作业 PDF 位于 `docs/pdf/最终大作业文档.pdf`。LaTeX 源文件位于 `docs/final_assignment_latex/main.tex`。
