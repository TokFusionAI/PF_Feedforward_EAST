"""paper_betan: 用最优消融模型 transformer_bidir_on (s44; 双向 Transformer + 物理时间正弦位置编码) 重画论文图 + test 评测 + FreeGSNKE 验证。

ckpt 自带 cfg (extra["cfg"]), 所以无需单独 yaml —— 模型/数据路径都从 ckpt 读。
"""
