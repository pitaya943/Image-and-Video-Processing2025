import numpy as np
import matplotlib.pyplot as plt
import cProfile

def segment_error_sum_vectorized(positions, i, j):
    """
    向量化計算從 positions[i] 到 positions[j] 之間所有中間點，
    與直線 (positions[i], positions[j]) 的距離平方和。

    如果中間無點 (即 j <= i + 1) 則回傳 0
    """
    if i + 1 >= j:
        return 0.0

    p0 = positions[i]
    p1 = positions[j]
    v = p1 - p0
    L2 = np.dot(v, v)
    if L2 == 0:
        return 0.0

    # 取得中間所有點 形狀為 (n, 2)
    pts = positions[i+1:j]
    # 計算每個點與 p0 的向量差
    diff = pts - p0
    # 計算外積：對於每個點 v[0]*(y - p0[1]) - v[1]*(x - p0[0])
    cross = v[0]*diff[:, 1] - v[1]*diff[:, 0]
    # 計算每個點的距離平方
    d2 = (cross**2) / L2
    # 回傳總和
    return np.sum(d2)

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
            error[i, j] = segment_error_sum_vectorized(points, i, j)
    
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

def main():
    # 產生一個測試曲線（例如圓形的離散點 但不強制閉合）
    x, y = generate_irregular_shape(1000, 10, 2, 10)
    points = np.column_stack((x, y))
    
    epsilon = 0.5  # 設定允許的誤差上限
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