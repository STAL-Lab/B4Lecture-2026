import pandas as pd
import numpy as np
import os

target_files = ['01.csv', '02.csv', '03.csv']

print(f"{'File':<10} | {'Ball Error':<12} | {'Left Team Error':<16} | {'Right Team Error':<17}")
print("-" * 65)

for file_name in target_files:
    pred = pd.read_csv(f'./results/test/submission/{file_name}')
    gt = pd.read_csv(f'./test_old/gt/{file_name}')
    inp = pd.read_csv(f'./test_old/input/{file_name}')
    
    start_idx = len(inp)
    gt_slice = gt.iloc[start_idx : start_idx + len(pred)].reset_index(drop=True)
    
    # ボールの誤差計算 
    b_err_seq = np.sqrt((pred['b_x'] - gt_slice['b_x'])**2 + (pred['b_y'] - gt_slice['b_y'])**2)
    b_mean_err = b_err_seq.mean()
    
    # 左チームの誤差計算
    l_cols_x = [c for c in pred.columns if c.startswith('l') and c.endswith('_x')]
    l_err_list = []
    max_l_err = 0
    max_l_player = ""
    max_l_frame = -1
    
    for cx in l_cols_x:
        cy = cx.replace('_x', '_y')
        err_seq = np.sqrt((pred[cx] - gt_slice[cx])**2 + (pred[cy] - gt_slice[cy])**2)
        l_err_list.append(err_seq.mean())
        
        if err_seq.max() > max_l_err:
            max_l_err = err_seq.max()
            max_l_player = cx.replace('_x', '')
            max_l_frame = err_seq.idxmax() + 1
            
    l_mean_err = np.mean(l_err_list)
    
    # 右チームの誤差計算
    r_cols_x = [c for c in pred.columns if c.startswith('r') and c.endswith('_x')]
    r_err_list = []
    max_r_err = 0
    max_r_player = ""
    max_r_frame = -1
    
    for cx in r_cols_x:
        cy = cx.replace('_x', '_y')
        err_seq = np.sqrt((pred[cx] - gt_slice[cx])**2 + (pred[cy] - gt_slice[cy])**2)
        r_err_list.append(err_seq.mean())
        
        if err_seq.max() > max_r_err:
            max_r_err = err_seq.max()
            max_r_player = cx.replace('_x', '')
            max_r_frame = err_seq.idxmax() + 1
            
    r_mean_err = np.mean(r_err_list)
    
    # 結果の出力
    print(f"{file_name:<10} | {b_mean_err:<12.4f} | {l_mean_err:<16.4f} | {r_mean_err:<17.4f}")
    
    # 各ファイルにおける最大誤差
    if b_err_seq.max() > max_l_err and b_err_seq.max() > max_r_err:
        print(f"  -> [最大誤差] 対象: ボール, 誤差: {b_err_seq.max():.4f}, フレーム: {b_err_seq.idxmax()+1}")
    elif max_l_err > max_r_err:
        print(f"  -> [最大誤差] 対象: 選手 {max_l_player}, 誤差: {max_l_err:.4f}, フレーム: {max_l_frame}")
    else:
        print(f"  -> [最大誤差] 対象: 選手 {max_r_player}, 誤差: {max_r_err:.4f}, フレーム: {max_r_frame}")
    print("-" * 65)