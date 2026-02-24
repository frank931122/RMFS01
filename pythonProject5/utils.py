# utils.py

def distance(point_a, point_b, *dicts):
    """
    计算两个点之间的曼哈顿距离。
    point_a, point_b 可以是:
      - (row, col) 形式的坐标元组
      - 整数位置编号(需要在传入的 dicts 里查找)

    *dicts: 一个或多个 映射 {int -> (row,col)} 的字典。
       - 如果 point_a 是 int, 我们会在这些 dict 中依次查询,
         找到则解码为坐标。

    返回: int/float, 表示 point_a 与 point_b 间的曼哈顿距离。
    """

    def decode_point(p):
        # 若 p 是 tuple 形式, 直接返回
        if isinstance(p, tuple) and len(p) == 2:
            return p
        # 若 p 是 int, 则在 dicts 中查找
        elif isinstance(p, int):
            for dct in dicts:
                if p in dct:
                    return dct[p]
            # 若都没找到
            raise KeyError(f"Location ID {p} not found in any of the provided dictionaries.")
        else:
            raise TypeError(f"Unsupported location type: {p}")

    a = decode_point(point_a)
    b = decode_point(point_b)
    # 曼哈顿距离
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
