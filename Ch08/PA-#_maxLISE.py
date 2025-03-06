import numpy as np
import matplotlib.pyplot as plt
import cProfile

def segment_error_max_vectorized(points, i, j):
    """
    計算從 points[i] 到 points[j] 之間的子序列
    與直線 (points[i], points[j]) 的最大垂直誤差
    
    如果中間無點 (即 j <= i + 1) 則回傳 0
    """
    if i + 1 >= j:
        return 0.0

    p_start = points[i]
    p_end = points[j]
    line_vec = p_end - p_start
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        return 0.0

    # 取得所有中間點 (i+1 到 j-1) 並計算它們相對於 p_start 的向量差
    pts = points[i+1:j]
    vecs = pts - p_start

    # 在 2D 中兩向量的外積可由 a[0]*b[1] - a[1]*b[0] 得到
    # 計算每個中間點與線段向量的外積的絕對值
    cross_vals = np.abs(vecs[:, 0] * line_vec[1] - vecs[:, 1] * line_vec[0])

    # 每個點的垂直距離為外積值除以線段長度
    distances = cross_vals / line_length

    # 返回最大垂直距離
    return np.max(distances)

def optimal_polygon_approximation(points, epsilon):
    """
    利用 DP 找出最少邊數的多邊形逼近
    使得原始點序列中，每段（由逼近直線表示）的最大誤差不超過 epsilon
    
    輸入:
        points: 形狀 (n, 2) 的 numpy 陣列 表示原始點序列（順序排列）
        epsilon: 誤差上限
    
    輸出:
        indices: 最終選取的頂點索引（包含首尾）
        approx_points: 對應的逼近多邊形頂點坐標
    """
    n = len(points)
    # 預先計算每一段的誤差：error[i, j] 表示從點 i 到 j 的段誤差
    error = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            error[i, j] = segment_error_max_vectorized(points, i, j)
    
    # dp[i] 表示從起點到點 i 所需的最少段數
    dp = [float('inf')] * n
    # prev[i] 用於記錄最佳分割時 點 i 之前的分割點
    prev = [-1] * n
    dp[0] = 0  # 起始點不需要段分割
    
    # 動態規劃：對每個點 i 嘗試從前面的各個點 j 接上段落
    # 並檢查從 j 到 i 的誤差是否滿足 epsilon 條件
    for i in range(1, n):
        for j in range(0, i):
            if error[j, i] <= epsilon:
                if dp[j] + 1 < dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
    
    # 回溯找出選取的頂點索引
    indices = []
    i = n - 1
    while i >= 0:
        indices.append(i)
        i = prev[i]
    indices.reverse()
    
    approx_points = points[indices]
    return indices, approx_points

def generate_irregular_shape(num_points, base_radius, noise_scale, smooth_factor):
    """
    生成一個不規則的形狀（例如地圖邊緣）
    
    參數:
      num_points: 邊緣上的點數（數值越大形狀越平滑）
      base_radius: 基準半徑
      noise_scale: 噪音尺度 用來控制半徑的隨機波動幅度
      smooth_factor: 平滑因子 數值越大平滑效果越明顯
    
    輸出:
      x, y: 形狀邊緣的 x 和 y 坐標
    """
    # 生成 0 到 2π 區間均勻分布的角度
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # 生成每個角度對應的隨機噪音 這裡使用標準正態分佈
    noise = np.random.randn(num_points) * noise_scale
    
    # 利用卷積進行平滑處理 減少噪音的突變
    kernel = np.ones(smooth_factor) / smooth_factor
    noise_smoothed = np.convolve(noise, kernel, mode='same')
    
    # 計算每個角度上的半徑：基準半徑加上平滑後的噪音
    radius = base_radius + noise_smoothed
    
    # 根據極座標轉換為笛卡爾座標
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    return x, y

# 測試範例
def main():
    # 產生一個測試曲線（例如圓形的離散點 但不強制閉合）
    x, y = generate_irregular_shape(1000, 10, 2, 10)
    points = np.column_stack((x, y))
    
    epsilon = 0.2  # 設定允許的誤差上限
    indices, approx_points = optimal_polygon_approximation(points, epsilon)
    print("PA-# vertices:", indices)
    print("PA-# vertices coordinate:\n", approx_points)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', label='Original Curve')
    plt.plot(approx_points[:, 0], approx_points[:, 1], 'ro-', linewidth=2, label='PA-#')
    plt.title(f"Polygonal Approximation (epsilon = {epsilon})")
    plt.legend()
    plt.show()

cProfile.run('main()')