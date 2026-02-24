import matplotlib.pyplot as plt

# 数据序列重新初始化
sequence = [f"n{i}" for i in range(1, 64)]
data = [sequence[i:i + 9] for i in range(0, len(sequence), 9)]

# 绘制表格
fig, ax = plt.subplots(figsize=(20, 20))  # 增加画布高度
ax.axis('tight')
ax.axis('off')

# 设置表格样式
table = ax.table(cellText=data, colLabels=[f"C{i}" for i in range(9)],
                 rowLabels=[f"R{i}" for i in range(7)], loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(22)  # 设置适合的字体大小
table.scale(1, 11)  # 大幅增加纵向比例

# 保存表格为图片
plt.savefig("high_table_image.png", bbox_inches='tight')
plt.close(fig)
