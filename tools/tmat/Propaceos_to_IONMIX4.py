#!/usr/bin/env python3
"""
PROPACEOS to IONMIX4 Converter

重要：IONMIX4形式では、不透明度のグリッド（温度/密度）とEOSのグリッドが等しい必要があります。
このコンバーターはPROPACEOSファイルからIONMIX4形式への変換を行います。
"""

import numpy as np
import matplotlib.pyplot as plt
import pprint
import linecache
import pandas as pd
import csv
import re

def pd_read_row(data, idx):
	"""指定されたインデックスの行からNaN値を除いた数値データを取得"""
	a = np.array(data.loc[idx])
	return a[~np.isnan(a)]

def output_str(data):
	"""数値データを科学記数法の文字列形式に変換し、4列ごとに改行を挿入"""
	a = ['{:.6E}'.format(n) for n in data.tolist()]
	for i in reversed(range(-(-len(data)//4))):
		a.insert(4*(i+1), '\n')
	return ''.join(a)


# ===== ファイル設定とデータ読み込み =====
path = "CD_2/CD_2.prp"
df = pd.read_csv(path, dtype=np.float64, sep='  ', header=None, skiprows=41, comment='*', engine='python')

# ===== 変数初期化 =====
Temp_points_EOS = 0    # EOS用温度グリッド点数
Dens_points_EOS = 0    # EOS用密度グリッド点数
Temp_points_Opa = 0    # Opacity用温度グリッド点数
Dens_points_Opa = 0    # Opacity用密度グリッド点数
Group_num = 0          # エネルギーグループ数
# ===== データ配列初期化 =====
emp_list =[]                    # 空のリスト（IONMIX4形式で必要）
Temp_list_EOS = []              # EOS用温度リスト
Dens_list_EOS = []              # EOS用密度リスト
Temp_list_Opa = []              # Opacity用温度リスト
Dens_list_Opa = []              # Opacity用密度リスト
Group_boundaries = []           # エネルギーグループ境界
Zbar_list = []                  # 平均イオン化度
Eion_list = []                  # イオン内部エネルギー
Eele_list = []                  # 電子内部エネルギー
Pion_list = []                  # イオン圧力
Pele_list = []                  # 電子圧力
Rosseland_Opacity = []          # Rosseland平均不透明度
Emission_Opacity = []           # 放射Planck平均不透明度
Absorption_Opacity = []         # 吸収Planck平均不透明度
Rosseland_Opacity_list = []     # IONMIX4形式用Rosseland不透明度
Emission_Opacity_list = []      # IONMIX4形式用放射不透明度
Absorption_Opacity_list = []    # IONMIX4形式用吸収不透明度


# ===== ヘッダー情報の読み取り =====
element_num = int(re.sub("\\D", "", linecache.getline(path, 20)))
# 31行目から原子番号のリストを抽出
line31 = linecache.getline(path, 31)
atomic_gases = [int(num) for num in re.findall(r'\d+', line31)]
relative_fractions = re.sub(" relative fractions:   ", "", linecache.getline(path, 34))

# ===== 温度・密度グリッドの読み取り =====
n = 0

# 温度点数の読み取り（41行目）
Temp_points_EOS = int(linecache.getline(path, n+41))

# 温度リストの読み取り（10個ずつの行に分割されている）
for i in range(-(-Temp_points_EOS//10)):
	Temp_list_EOS = np.hstack((Temp_list_EOS, pd_read_row(df,n+i)))

n = n + -(-Temp_points_EOS//10)

# 密度点数の読み取り
Dens_points_EOS = int(pd_read_row(df, n).item())

n = n+1

# 密度リストの読み取り（10個ずつの行に分割されている）
for i in range(-(-Dens_points_EOS//10)):
	Dens_list_EOS = np.hstack((Dens_list_EOS, pd_read_row(df,n+i)))

# データの位置をスキップ（中間データを飛ばす）
n = n + -(-Dens_points_EOS//10) + 1

Temp_points_Opa = int(pd_read_row(df, n).item())

n = n+1

# 温度リストの読み取り（10個ずつの行に分割されている）
for i in range(-(-Temp_points_Opa//10)):
	Temp_list_Opa = np.hstack((Temp_list_Opa, pd_read_row(df,n+i)))

n = n + -(-Temp_points_Opa//10)

# 密度点数の読み取り
Dens_points_Opa = int(pd_read_row(df, n).item())

n = n+1

# 密度リストの読み取り（10個ずつの行に分割されている）
for i in range(-(-Dens_points_Opa//10)):
	Dens_list_Opa = np.hstack((Dens_list_Opa, pd_read_row(df,n+i)))

# データの位置をスキップ（中間データを飛ばす）
n = n + -(-Dens_points_Opa//10)

# ===== グリッド一致性チェック =====
# IONMIX4形式では、不透明度のグリッドとEOSのグリッドが等しい必要がある
if Temp_points_EOS != Temp_points_Opa:
    print(f"エラー: 温度点数が一致しません (EOS: {Temp_points_EOS}, Opacity: {Temp_points_Opa})")
    exit(1)

if Dens_points_EOS != Dens_points_Opa:
    print(f"エラー: 密度点数が一致しません (EOS: {Dens_points_EOS}, Opacity: {Dens_points_Opa})")
    exit(1)

# 温度リストの一致性をチェック
if not np.allclose(Temp_list_EOS, Temp_list_Opa, rtol=1e-10):
    print("エラー: 温度リストが一致しません")
    print(f"EOS温度リスト: {Temp_list_EOS[:5]}...")
    print(f"Opacity温度リスト: {Temp_list_Opa[:5]}...")
    exit(1)

# 密度リストの一致性をチェック
if not np.allclose(Dens_list_EOS, Dens_list_Opa, rtol=1e-10):
    print("エラー: 密度リストが一致しません")
    print(f"EOS密度リスト: {Dens_list_EOS[:5]}...")
    print(f"Opacity密度リスト: {Dens_list_Opa[:5]}...")
    exit(1)

print("グリッド一致性チェック: OK")



# 空のリストを初期化（温度×密度点数分）
emp_list = np.zeros(Temp_points_EOS*Dens_points_EOS)

# エネルギーグループ数の読み取り
Group_num = int(pd_read_row(df, n).item())

n = n+1

# エネルギーグループ境界の読み取り
for i in range(-(-Group_num//10)+1):
	Group_boundaries = np.hstack((Group_boundaries, pd_read_row(df,n+i)))

# ===== データセクションへの位置調整 =====
n = n + -((-Group_num-1)//10) 

# 各原子ガスの分のデータをスキップ
for i in range(len(atomic_gases)):
	n = n + -(-(Temp_points_EOS*Dens_points_EOS*(atomic_gases[i]+1))//10)

# ===== EOS（状態方程式）データの読み取り =====

# 平均イオン化度（z_bar）の読み取り
for i in range(-(-Temp_points_EOS*Dens_points_EOS//10)):
	Zbar_list = np.hstack((Zbar_list, pd_read_row(df,n+i)))

n = n + -(-Temp_points_EOS*Dens_points_EOS//10)

# 統合不透明度データをスキップ（Int. Rosseland, Int. emission, Int. absorption, Eint）
# 各温度/密度点につき4つのデータがある
n = n + -(-Temp_points_Opa*Dens_points_Opa//10)*3

# イオン比内部エネルギー Eion (J/g)
for i in range(-(-Temp_points_EOS*Dens_points_EOS//10)):
	Eion_list = np.hstack((Eion_list, pd_read_row(df,n+i)))

n = n + -(-Temp_points_EOS*Dens_points_EOS//10)

# 電子比内部エネルギー Eele (J/g)
for i in range(-(-Temp_points_EOS*Dens_points_EOS//10)):
	Eele_list = np.hstack((Eele_list, pd_read_row(df,n+i)))

n = n + -(-Temp_points_EOS*Dens_points_EOS//10)

# イオン圧力 Pion (dyne/cm**2) → (J/cm3)への変換
for i in range(-(-Temp_points_EOS*Dens_points_EOS//10)):
	Pion_list = np.hstack((Pion_list, pd_read_row(df,n+i)))
Pion_list = Pion_list * 1.0E-07  # 単位変換

n = n + -(-Temp_points_EOS*Dens_points_EOS//10)

# 電子圧力 Pele (dyne/cm**2) → (J/cm3)への変換
for i in range(-(-Temp_points_EOS*Dens_points_EOS//10)):
	Pele_list = np.hstack((Pele_list, pd_read_row(df,n+i)))
Pele_list = Pele_list * 1.0E-07  # 単位変換

n = n + -(-Temp_points_EOS*Dens_points_EOS//10)

# ===== 多群不透明度データの読み取り =====
# IONMIX4形式では、不透明度のグリッド（温度/密度）とEOSのグリッドが等しい必要がある

# 全温度/密度点の不透明度データを読み取り
total_td_points = Temp_points_Opa * Dens_points_Opa

for i in range(total_td_points):
	# Rosseland平均不透明度
	tmp = []
	for j in range(-(-Group_num//10)):
		tmp = np.hstack([tmp, pd_read_row(df,n+j)])
	
	if i == 0:
		Rosseland_Opacity = tmp
	else:
		Rosseland_Opacity = np.vstack([Rosseland_Opacity, tmp])
	n = n + -(-Group_num//10)

	# 放射Planck平均不透明度
	tmp = []
	for j in range(-(-Group_num//10)):
		tmp = np.hstack([tmp, pd_read_row(df,n+j)])
	
	if i == 0:
		Emission_Opacity = tmp
	else:
		Emission_Opacity = np.vstack([Emission_Opacity, tmp])
	n = n + -(-Group_num//10)

	# 吸収Planck平均不透明度
	tmp = []
	for j in range(-(-Group_num//10)):
		tmp = np.hstack([tmp, pd_read_row(df,n+j)])
	
	if i == 0:
		Absorption_Opacity = tmp
	else:
		Absorption_Opacity = np.vstack([Absorption_Opacity, tmp])
	n = n + -(-Group_num//10)

# ===== IONMIX4形式用データの整理 =====
# グループごとに温度/密度データを並び替え
for i in range(Group_num):
	Rosseland_Opacity_list = np.hstack([Rosseland_Opacity_list, Rosseland_Opacity[:,i]])
	Emission_Opacity_list = np.hstack([Emission_Opacity_list, Emission_Opacity[:,i]])
	Absorption_Opacity_list = np.hstack([Absorption_Opacity_list, Absorption_Opacity[:,i]])

# ===== IONMIX4ファイルの出力 =====
with open('{}.cn4'.format(path[:len(path)-4]), 'w') as f:
	# ヘッダー行1: 温度点数と密度点数
	f.write('        {0}        {1}\n'.format(Temp_points_EOS, Dens_points_EOS))
	# ヘッダー行2: 原子番号（スペース7個間隔）
	f.write(' atomic #s of gases:       {}\n'.format('       '.join(map(str, atomic_gases))))
	# ヘッダー行3: 相対分数
	f.write(' relative fractions:   {}'.format(relative_fractions))
	# ヘッダー行4: エネルギーグループ数
	f.write('          {}\n'.format(Group_num))
	
	# データセクション（IONMIX4形式の順序）
	f.write(output_str(Temp_list_EOS))              # 温度リスト
	f.write(output_str(Dens_list_EOS))              # 密度リスト
	f.write(output_str(Zbar_list))              # 平均イオン化度
	f.write(output_str(emp_list))               # 空のリスト1
	f.write(output_str(Pion_list))              # イオン圧力
	f.write(output_str(Pele_list))              # 電子圧力
	f.write(output_str(emp_list))               # 空のリスト2
	f.write(output_str(emp_list))               # 空のリスト3
	f.write(output_str(Eion_list))              # イオン内部エネルギー
	f.write(output_str(Eele_list))              # 電子内部エネルギー
	f.write(output_str(emp_list))               # 空のリスト4
	f.write(output_str(emp_list))               # 空のリスト5
	f.write(output_str(emp_list))               # 空のリスト6
	f.write(output_str(emp_list))               # 空のリスト7
	f.write(output_str(Group_boundaries))       # エネルギーグループ境界
	f.write(output_str(Rosseland_Opacity_list)) # Rosseland不透明度
	f.write(output_str(Absorption_Opacity_list))# 吸収不透明度
	f.write(output_str(Emission_Opacity_list))  # 放射不透明度

pprint.pprint('Success')

